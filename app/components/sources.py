# app/components/sources.py


import streamlit as st


def render_sources(chunks: list[dict], expanded: bool = False) -> None:
    
    if not chunks:
        return

    with st.expander(
        f"📄 View retrieved sources ({len(chunks)} chunks)",
        expanded=expanded,
    ):
        for i, chunk in enumerate(chunks, start=1):
            meta    = chunk.get("metadata", {})
            source  = meta.get("source_file", "unknown")
            page    = meta.get("page_num", "?")
            section = meta.get("section_heading", "")
            score   = chunk.get("rerank_score")
            text    = chunk.get("text", "")

            col1, col2 = st.columns([3, 1])
            with col1:
                label = f"**[{i}]** `{source}` — Page {page}"
                if section:
                    label += f" · *{section}*"
                st.markdown(label)
            with col2:
                if score is not None:
                    # color code by score
                    color = "green" if score > 3.0 else "orange" if score > 0 else "red"
                    st.markdown(
                        f":{color}[Score: {score:.3f}]"
                    )

            st.caption(text[:400] + ("..." if len(text) > 400 else ""))

            if i < len(chunks):
                st.divider()


def render_retrieval_debug(result: dict) -> None:
    
    d = len(result.get("dense_results", []))
    s = len(result.get("sparse_results", []))
    f = len(result.get("fused_results", []))
    r = len(result.get("final_results", []))
    st.caption(
        f"🔍 Pipeline: {d} dense + {s} sparse → {f} fused (RRF) → {r} reranked"
    )