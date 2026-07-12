# app/streamlit_app.py


import sys
import streamlit as st
from pathlib import Path


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


st.set_page_config(
    page_title="Depth Estimation Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


from app.utils.loaders import load_pipeline, load_llm_client, load_callables
from app.components.sidebar import render_sidebar
from app.components.chat import (
    init_session_state,
    render_chat_history,
    add_user_message,
    add_assistant_message,
)
from app.components.query_handler import handle_query


def main():

    init_session_state()

    
    with st.spinner("🔬 Loading research assistant (first load: ~30-60s on CPU)..."):
        pipeline   = load_pipeline()
        llm_client = load_llm_client()
        callables  = load_callables()

    
    settings = render_sidebar()

    
    st.title("🔬 Depth Estimation Research Assistant")
    st.caption(
        "Ask questions about monocular depth estimation, stereo methods, "
        "and related computer vision research — grounded in indexed papers."
    )

    
    render_chat_history(show_sources=settings["show_sources"])

    
    typed_query = st.chat_input(
        "Ask about depth estimation research...",
        disabled=st.session_state.processing,
    )
    query = typed_query or settings["selected_example"]

    if not query:
        if not st.session_state.messages:
            st.info(
                "👆 Type a question or click an example in the sidebar.",
                icon="💡",
            )
        return

    if st.session_state.processing:
        return

    
    add_user_message(query)

    st.session_state.processing = True
    try:
        retrieval_result, answer = handle_query(
            query, pipeline, llm_client, callables, settings
        )
    finally:
        st.session_state.processing = False

    
    sources = (retrieval_result or {}).get("final_results", [])
    add_assistant_message(answer, sources)


if __name__ == "__main__":
    main()