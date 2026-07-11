# app/streamlit_app.py
"""
Streamlit UI for the FYP Research Assistant.

Design principles:
- Show answer + sources + provenance on every query
- Load models once via @st.cache_resource
- Split retrieval and generation so sources appear before the answer
- Stream LLM response for perceived speed
- Graceful 429 fallback — show retrieved sources even if LLM fails
"""

import streamlit as st
import time
import sys
from pathlib import Path

# ── make src/ importable when running from project root ───────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import RAGPipeline
from src.generation.prompt_templates import build_rag_prompt
from src.generation.llm_client import LLMClient

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Depth Estimation Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── example questions from your eval set ─────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "What is the main limitation of monocular depth estimation?",
    "How does unsupervised depth estimation avoid ground truth depth data?",
    "How does monocular depth estimation compare to stereo depth estimation?",
    "What visual cues do traditional handcrafted methods use for depth estimation?",
    "What are the five main challenges for future monocular depth estimation?",
]

# ── corpus stats (update if you add more papers) ──────────────────────────────
CORPUS_STATS = {
    "papers": 8,
    "chunks": 611,
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "retrieval": "Hybrid (BM25 + Dense + Rerank)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loading — runs ONCE per session, not on every rerun
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading research assistant (first load takes ~30s)...")
def get_pipeline() -> RAGPipeline:
    """
    Load RAGPipeline once. Without @st.cache_resource, Streamlit reruns
    this on every interaction, reloading BGE + BM25 + cross-encoder each time.
    """
    return RAGPipeline()


@st.cache_resource(show_spinner=False)
def get_llm_client() -> LLMClient:
    return LLMClient()


# ─────────────────────────────────────────────────────────────────────────────
# Streaming generation — renders tokens as they arrive
# ─────────────────────────────────────────────────────────────────────────────

def generate_streaming(llm_client: LLMClient, messages: list[dict]):
    """
    Generator that yields text chunks from the LLM stream.
    Used with st.write_stream() for token-by-token rendering.
    Yields full response as fallback if streaming isn't supported.
    """
    try:
        import os
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        stream = client.chat.completions.create(
            model=llm_client.model,
            messages=messages,
            max_tokens=llm_client.max_tokens,
            temperature=llm_client.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception:
        # fallback to non-streaming if anything fails
        response = llm_client.generate(messages)
        yield response


def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(t in msg for t in ["429", "rate limit", "rate-limit", "too many requests"])


# ─────────────────────────────────────────────────────────────────────────────
# Source display helpers
# ─────────────────────────────────────────────────────────────────────────────

def render_sources(chunks: list[dict], show_scores: bool = True) -> None:
    """Render retrieved chunks in an expandable section."""
    with st.expander(f"📄 View retrieved sources ({len(chunks)} chunks)", expanded=False):
        for i, chunk in enumerate(chunks, start=1):
            meta      = chunk.get("metadata", {})
            source    = meta.get("source_file", "unknown")
            page      = meta.get("page_num", "?")
            section   = meta.get("section_heading", "")
            score     = chunk.get("rerank_score", None)
            text      = chunk.get("text", "")

            col1, col2 = st.columns([3, 1])
            with col1:
                label = f"**[{i}]** `{source}` — Page {page}"
                if section:
                    label += f" · *{section}*"
                st.markdown(label)
            with col2:
                if show_scores and score is not None:
                    st.caption(f"Rerank score: {score:.3f}")

            st.caption(text[:400] + ("..." if len(text) > 400 else ""))

            if i < len(chunks):
                st.divider()


def render_retrieval_debug(result: dict) -> None:
    """Show dense/sparse/fused candidate counts for transparency."""
    d = len(result.get("dense_results", []))
    s = len(result.get("sparse_results", []))
    f = len(result.get("fused_results", []))
    r = len(result.get("final_results", []))
    st.caption(
        f"Retrieval: {d} dense + {s} sparse → {f} fused → {r} reranked"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar and return user settings dict."""
    with st.sidebar:
        st.title("🔬 Research Assistant")
        st.caption("Depth Estimation Paper Q&A")
        st.divider()

        # ── corpus stats ──────────────────────────────────────────────────────
        st.subheader("📚 Corpus")
        st.markdown(f"""
        - **{CORPUS_STATS['papers']} papers** indexed
        - **{CORPUS_STATS['chunks']} chunks** total
        - **Retrieval:** {CORPUS_STATS['retrieval']}
        - **Embeddings:** `{CORPUS_STATS['embedding_model']}`
        """)
        st.divider()

        # ── settings ──────────────────────────────────────────────────────────
        st.subheader("⚙️ Settings")

        show_sources = st.toggle(
            "Show retrieved sources",
            value=True,
            help="Display the chunks used to generate each answer"
        )
        retrieval_only = st.toggle(
            "Retrieval-only mode",
            value=False,
            help="Show retrieved chunks without calling the LLM — useful when rate limited"
        )

        with st.expander("Advanced"):
            final_top_k = st.slider(
                "Chunks sent to LLM",
                min_value=3,
                max_value=7,
                value=5,
                help="More chunks = more context but slower and more expensive"
            )

        st.divider()

        # ── example questions ─────────────────────────────────────────────────
        st.subheader("💡 Example questions")
        st.caption("Click to ask")

        selected_example = None
        for q in EXAMPLE_QUESTIONS:
            if st.button(q[:60] + ("..." if len(q) > 60 else ""), use_container_width=True):
                selected_example = q

        st.divider()

        # ── clear history ─────────────────────────────────────────────────────
        if st.button("🗑️ Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return {
        "show_sources":   show_sources,
        "retrieval_only": retrieval_only,
        "final_top_k":    final_top_k,
        "selected_example": selected_example,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main query handler
# ─────────────────────────────────────────────────────────────────────────────

def handle_query(
    query: str,
    pipeline: RAGPipeline,
    llm_client: LLMClient,
    settings: dict,
) -> None:
    """
    Run the full pipeline for one query and render the assistant turn.
    Splits retrieval and generation so sources appear immediately.
    """
    with st.chat_message("assistant"):

        # ── step 1-4: retrieval with progress ─────────────────────────────────
        retrieval_result = None
        with st.status("Searching research papers...", expanded=True) as status:
            st.write("⏳ Embedding query...")
            t0 = time.time()

            st.write("⏳ Searching dense + sparse index...")
            dense_results  = pipeline.dense_retriever.retrieve(
                query, top_k=15
            )
            sparse_results = pipeline.sparse_retriever.retrieve(
                query, top_k=15
            )

            st.write("⏳ Fusing and reranking candidates...")
            from src.retrieval.fusion import reciprocal_rank_fusion
            fused   = reciprocal_rank_fusion(dense_results, sparse_results)
            reranked = pipeline.reranker.rerank(
                query, fused[:15], top_k=settings["final_top_k"]
            )

            retrieval_result = {
                "query":          query,
                "dense_results":  dense_results,
                "sparse_results": sparse_results,
                "fused_results":  fused,
                "final_results":  reranked,
            }

            elapsed = time.time() - t0
            status.update(
                label=f"✅ Retrieved {len(reranked)} chunks in {elapsed:.1f}s",
                state="complete",
                expanded=False,
            )

        # ── show sources immediately (before LLM call) ────────────────────────
        if settings["show_sources"]:
            render_sources(reranked)
            render_retrieval_debug(retrieval_result)

        # ── retrieval-only mode — stop here ───────────────────────────────────
        if settings["retrieval_only"]:
            st.info(
                "ℹ️ Retrieval-only mode — LLM generation skipped. "
                "Toggle off in sidebar to get a full answer."
            )
            return retrieval_result, None

        # ── step 5: generate answer (streaming) ───────────────────────────────
        answer = ""
        try:
            messages = build_rag_prompt(query, reranked)

            st.write("**Answer:**")
            with st.spinner("Generating answer..."):
                answer = st.write_stream(
                    generate_streaming(llm_client, messages)
                )

        except Exception as e:
            if is_rate_limit_error(e):
                st.warning(
                    "⚠️ The language model is temporarily rate-limited. "
                    "Retrieved sources are shown above — try again in 30 seconds.",
                    icon="⏱️"
                )
            else:
                st.error(f"Generation failed: {e}")
            answer = None

    return retrieval_result, answer


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── session state init ────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # ── load models ───────────────────────────────────────────────────────────
    pipeline   = get_pipeline()
    llm_client = get_llm_client()

    # ── sidebar ───────────────────────────────────────────────────────────────
    settings = render_sidebar()

    # ── main area header ──────────────────────────────────────────────────────
    st.title("🔬 Depth Estimation Research Assistant")
    st.caption(
        "Ask questions about monocular depth estimation, stereo methods, "
        "and related computer vision research — grounded in indexed papers."
    )

    # ── render chat history ───────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources") and settings["show_sources"]:
                render_sources(msg["sources"])

    # ── determine query — typed or example click ──────────────────────────────
    typed_query   = st.chat_input(
        "Ask about depth estimation research...",
        disabled=st.session_state.processing,
    )
    query = typed_query or settings["selected_example"]

    if not query:
        # ── empty state ───────────────────────────────────────────────────────
        if not st.session_state.messages:
            st.info(
                "👆 Type a question or click an example in the sidebar to get started.",
                icon="💡"
            )
        return

    # ── guard against double submit ───────────────────────────────────────────
    if st.session_state.processing:
        return

    # ── add user message to history ───────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── run pipeline ──────────────────────────────────────────────────────────
    st.session_state.processing = True
    try:
        retrieval_result, answer = handle_query(
            query, pipeline, llm_client, settings
        )
    finally:
        st.session_state.processing = False

    # ── save assistant turn to history ────────────────────────────────────────
    sources = retrieval_result["final_results"] if retrieval_result else []
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer or "_Sources retrieved above — generation unavailable._",
        "sources": sources,
    })


if __name__ == "__main__":
    main()