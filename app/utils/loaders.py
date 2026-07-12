# app/utils/loaders.py

import streamlit as st


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """
    Load RAGPipeline once — BGE embedder + BM25 index + cross-encoder.
    On your i5-5300U CPU this takes ~30-60s on first load.
    After that, every call returns the same cached object instantly.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from src.pipeline import RAGPipeline
    return RAGPipeline()


@st.cache_resource(show_spinner=False)
def load_llm_client():
    """Load LLMClient once — just reads API key and creates OpenAI client."""
    from src.generation.llm_client import LLMClient
    return LLMClient()


@st.cache_resource(show_spinner=False)
def load_callables() -> dict:
    """
    Pre-import and return callables needed at query time.
    Avoids re-importing on every query which adds latency.
    """
    from src.generation.prompt_templates import build_rag_prompt
    from src.retrieval.fusion import reciprocal_rank_fusion
    return {
        "build_rag_prompt":        build_rag_prompt,
        "reciprocal_rank_fusion":  reciprocal_rank_fusion,
    }