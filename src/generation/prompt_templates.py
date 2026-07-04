# src/generation/prompt_templates.py
"""
prompt_templates.py — builds the prompt handed to the LLM.

The prompt design here makes three explicit demands on the LLM:
1. Answer ONLY from the provided context chunks — not from general knowledge.
2. Cite every claim with [source_file, page N] — so faithfulness is auditable.
3. Explicitly say "not found in context" if the answer isn't supported.

Demand #3 is the most important one for RAG correctness: without it, the
LLM will hallucinate a plausible-sounding answer even when the retrieved
chunks don't actually support one. This is RAG's most common failure mode
in production, and the prompt is your primary defense against it.
"""

from src.retrieval.reranker import Reranker  # only for type hint below


def format_context_block(chunks: list[dict]) -> str:
    """
    Serializes the top-k reranked chunks into a numbered context block
    the LLM can reference. Each chunk gets a [N] label so the LLM can
    cite it by number in the answer, alongside source/page metadata.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"].get("source_file", "unknown")
        page = chunk["metadata"].get("page_num", "?")
        section = chunk["metadata"].get("section_heading", "")
        header = f"[{i}] Source: {source}, Page {page}"
        if section:
            header += f", Section: {section}"
        lines.append(header)
        lines.append(chunk["text"])
        lines.append("")  # blank line between chunks
    return "\n".join(lines)


def build_rag_prompt(query: str, chunks: list[dict]) -> list[dict]:
    """
    Returns a messages list (OpenAI/OpenRouter format) ready to pass
    directly to the LLM client.

    System message: establishes the rules of engagement (context-only,
    must cite, must admit absence of answer).
    User message: the actual query + the formatted context block.

    Keeping them separate (system vs user) rather than cramming everything
    into one user message gives the model clearer role separation and
    generally produces more faithful, citation-structured outputs.
    """
    system_prompt = """You are a research assistant for a computer vision \
    research project on depth estimation.

    You will be given a question and a set of numbered context chunks retrieved \
    from academic papers. Your job is to answer the question based ONLY on the \
    provided context.

    Rules you must follow strictly:
    1. Answer only from the provided context chunks. Do not use outside knowledge.
    2. Cite every factual claim using [N] notation where N is the chunk number.
    Include the source file and page at the end of your answer in a References section.
    3. If the context chunks do not contain enough information to answer the question,
    say exactly: "The provided context does not contain sufficient information \
    to answer this question." Do not guess or fill in from general knowledge.
    4. Keep your answer focused and concise. Do not pad with unnecessary background.
    """

    context_block = format_context_block(chunks)

    user_message = f"""Question: {query}

    Context:
    {context_block}

    Answer (with citations):"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]