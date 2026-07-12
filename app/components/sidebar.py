# app/components/sidebar.py


import streamlit as st
from app.utils.helper import EXAMPLE_QUESTIONS, CORPUS_STATS


def render_sidebar() -> dict:
    """
    Renders the full sidebar and returns user settings.

    Returns:
        dict with keys:
            show_sources    — bool, display retrieved chunks
            retrieval_only  — bool, skip LLM generation
            final_top_k     — int, chunks sent to LLM (3-7)
            selected_example — str | None, clicked example question
    """
    with st.sidebar:
        st.title("🔬 Research Assistant")
        st.caption("Depth Estimation Paper Q&A")
        st.divider()

        
        st.subheader("📚 Corpus")
        st.markdown(f"""
            - **{CORPUS_STATS['papers']} papers** indexed
            - **{CORPUS_STATS['chunks']} chunks** total
            - **Retrieval:** {CORPUS_STATS['retrieval']}
            - **Embeddings:** `{CORPUS_STATS['embedding_model']}`
        """)
        st.divider()

        
        st.subheader("⚙️ Settings")

        show_sources = st.toggle(
            "Show retrieved sources",
            value=True,
            help="Display the paper chunks used to generate each answer"
        )
        retrieval_only = st.toggle(
            "Retrieval-only mode",
            value=False,
            help="Skip LLM generation — useful when rate limited"
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

        
        st.subheader("💡 Example questions")
        st.caption("Click any question to ask it")

        selected_example = None
        for q in EXAMPLE_QUESTIONS:
            label = q[:55] + ("..." if len(q) > 55 else "")
            if st.button(
                label,
                use_container_width=True,
                key=f"example__{q[:25]}",
            ):
                selected_example = q

        st.divider()

        
        if st.button("🗑️ Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        
        with st.expander("📊 Hybrid RAG Eval results"):
            st.markdown("""
                **Retrieval (25 questions)**
                | Config | Recall@5 | MRR |
                |---|---|---|
                | Dense only | 0.60 | 0.496 |
                | Sparse only | 0.40 | 0.296 |
                | **Hybrid** | **0.76** | **0.555** |

                **Generation (RAGAS)**
                | Metric | Score |
                |---|---|
                | Faithfulness | 0.824 |
                | Answer Relevancy | 0.888 |
            """)

    return {
        "show_sources":     show_sources,
        "retrieval_only":   retrieval_only,
        "final_top_k":      final_top_k,
        "selected_example": selected_example,
    }