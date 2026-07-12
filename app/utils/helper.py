# app/utils/helpers.py


import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def get_api_key(key_name: str) -> str:
    """
    Read API key from Streamlit secrets (cloud) or .env (local).
    Streamlit Cloud injects secrets via st.secrets automatically.
    Falls back to environment variable for local development.
    """
    try:
        return st.secrets[key_name]
    except Exception:
        return os.getenv(key_name, "")


def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(t in msg for t in [
        "429", "rate limit", "rate-limit", "too many requests"
    ])



EXAMPLE_QUESTIONS = [
    "What is the main limitation of monocular depth estimation?",
    "How does unsupervised depth estimation avoid ground truth depth data?",
    "How does monocular depth estimation compare to stereo depth estimation?",
    "What visual cues do traditional handcrafted methods use for depth estimation?",
    "What are the five main challenges for future monocular depth estimation?",
]

CORPUS_STATS = {
    "papers":          8,
    "chunks":          611,
    "embedding_model": "BAAI/bge-base-en-v1.5",
    "retrieval":       "Hybrid (BM25 + Dense + Rerank)",
}