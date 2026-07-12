# app/components/chat.py


import streamlit as st
from app.components.sources import render_sources


def init_session_state() -> None:
    """Initialize session state keys if they don't exist yet."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False


def render_chat_history(show_sources: bool = True) -> None:
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if (
                msg["role"] == "assistant"
                and msg.get("sources")
                and show_sources
            ):
                render_sources(msg["sources"])


def add_user_message(query: str) -> None:
    """Add a user turn to history and render it immediately."""
    st.session_state.messages.append({
        "role":    "user",
        "content": query,
    })
    with st.chat_message("user"):
        st.markdown(query)


def add_assistant_message(answer: str | None, sources: list[dict]) -> None:
    """Save the assistant turn to history for replay on next rerun."""
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer or "_Sources retrieved — generation unavailable._",
        "sources": sources,
    })