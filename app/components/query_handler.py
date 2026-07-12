# app/components/query_handler.py
"""
Query execution — runs the full RAG pipeline for one user question.

Deliberately splits retrieval and generation into two phases:
  Phase 1 (retrieval): ~2-4s — shows sources immediately
  Phase 2 (generation): ~5-15s — streams answer token by token

This makes the app feel responsive even when LLM calls are slow,
because users see retrieved chunks while the answer is still generating.
"""

import time
import streamlit as st

from app.components.sources import render_sources, render_retrieval_debug
from app.utils.helper import is_rate_limit_error, get_api_key


def generate_streaming(llm_client, messages: list[dict]):
    """
    Generator that yields LLM response tokens one by one.
    Used with st.write_stream() for perceived speed improvement —
    users see words appear as they're generated rather than waiting
    for the full response before anything is shown.

    Falls back to non-streaming if streaming fails.
    """
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=get_api_key("OPENROUTER_API_KEY"),
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
        # non-streaming fallback
        response = llm_client.generate(messages)
        yield response


def handle_query(
    query: str,
    pipeline,
    llm_client,
    callables: dict,
    settings: dict,
) -> tuple[dict | None, str | None]:
    
    build_rag_prompt       = callables["build_rag_prompt"]
    reciprocal_rank_fusion = callables["reciprocal_rank_fusion"]

    retrieval_result = None
    answer           = None

    with st.chat_message("assistant"):

        
        with st.status("Searching research papers...", expanded=True) as status:
            st.write("⏳ Embedding query...")
            t0 = time.time()

            st.write("⏳ Searching dense + sparse index...")
            dense_results  = pipeline.dense_retriever.retrieve(query, top_k=15)
            sparse_results = pipeline.sparse_retriever.retrieve(query, top_k=15)

            st.write("⏳ Fusing and reranking candidates...")
            fused    = reciprocal_rank_fusion(dense_results, sparse_results)
            reranked = pipeline.reranker.rerank(
                query,
                fused[:15],
                top_k=settings["final_top_k"],
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

        
        if settings["show_sources"]:
            render_sources(reranked)
            render_retrieval_debug(retrieval_result)

        
        if settings["retrieval_only"]:
            st.info(
                "ℹ️ Retrieval-only mode — toggle off in sidebar to generate an answer."
            )
            return retrieval_result, None

        
        try:
            messages = build_rag_prompt(query, reranked)
            st.markdown("**Answer:**")
            answer = st.write_stream(
                generate_streaming(llm_client, messages)
            )

        except Exception as e:
            if is_rate_limit_error(e):
                st.warning(
                    "⚠️ The language model is temporarily rate-limited. "
                    "Retrieved sources are shown above — try again in 30 seconds.",
                    icon="⏱️",
                )
            else:
                st.error(f"Generation failed: {e}")

    return retrieval_result, answer