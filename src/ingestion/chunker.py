# src/ingestion/chunker.py

import re
from dataclasses import dataclass
from typing import List
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


@dataclass
class DocumentChunk:
    text: str
    metadata: dict


class ProductionLangChainChunker:
    """
    Chunks PAGE BY PAGE instead of merging all pages into one giant string.

    Why: the previous approach merged everything into one stream, then tried
    to re-find each chunk's text inside that stream to recover its page
    number. That fails because LangChain's splitters rewrite whitespace when
    reconstructing page_content (collapsing "\\n\\n" into "  \\n", etc.), so
    chunk_text never byte-matches the original stream and .find() returns -1
    every time -- silently defaulting every chunk to page 1.

    Chunking per page sidesteps the problem entirely: we always know which
    page we're processing, because we're never out of that page's scope.
    The only thing we lose is letting a section span cleanly across a page
    break without a chunk boundary there too -- an acceptable tradeoff for
    correct metadata, and noted below if you want to revisit it later.
    """

    def __init__(self, chunk_size_chars: int = 1500, chunk_overlap_chars: int = 250):
        self.chunk_size = chunk_size_chars
        self.chunk_overlap = chunk_overlap_chars

        headers_to_split_on = [
            ("#", "Header_1"),
            ("##", "Header_2"),
            ("###", "Header_3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk_document(self, parsed_doc) -> List[DocumentChunk]:
        final_chunks = []

        # Track the most recently seen heading ACROSS pages, so that a page
        # with no heading of its own (e.g. body text continuing from the
        # previous page) still inherits the correct section name instead of
        # falling back to a generic placeholder.
        last_known_heading = "Abstract / Introduction"

        for page in parsed_doc.pages:
            page_text = page.markdown_text
            if not page_text.strip():
                continue

            # Structural phase: find headings WITHIN this single page.
            sections = self.markdown_splitter.split_text(page_text)

            # Refinement phase: break each section into RAG-sized chunks.
            sub_chunks = self.text_splitter.split_documents(sections)

            for chunk in sub_chunks:
                chunk_text = chunk.page_content

                h1 = chunk.metadata.get("Header_1", "")
                h2 = chunk.metadata.get("Header_2", "")
                h3 = chunk.metadata.get("Header_3", "")
                raw_heading = h1 or h2 or h3

                if raw_heading:
                    clean_heading = re.sub(r"[#\*_]", "", raw_heading).strip()
                    last_known_heading = clean_heading
                else:
                    clean_heading = last_known_heading

                # page.page_num comes directly from the page we're iterating --
                # no re-searching, no off-by-one risk, always correct.
                final_chunks.append(DocumentChunk(
                    text=chunk_text,
                    metadata={
                        "source_file": parsed_doc.source_file,
                        "page_num": page.page_num,
                        "section_heading": clean_heading,
                    }
                ))

        return final_chunks

    def chunk_all_documents(self, parsed_docs: List) -> List[DocumentChunk]:
        """Chunk a LIST of ParsedDocuments (one per PDF). This is what the
        ingestion orchestrator (build_index.py) should call."""
        all_chunks = []
        for doc in parsed_docs:
            doc_chunks = self.chunk_document(doc)
            print(f"  {doc.source_file}: {len(doc_chunks)} chunks")
            all_chunks.extend(doc_chunks)
        return all_chunks


if __name__ == "__main__":
    from src.ingestion.parser import parse_pdf
    from pathlib import Path

    sample_path = Path("data/raw/1901.09402v1 (1).pdf")

    if sample_path.exists():
        print("--- Testing per-page LangChain chunker ---")
        parsed_data = parse_pdf(sample_path)

        chunker = ProductionLangChainChunker(chunk_size_chars=1200, chunk_overlap_chars=200)
        chunks = chunker.chunk_document(parsed_data)

        print(f"Total Chunks Generated: {len(chunks)}")

        page_nums_seen = sorted(set(c.metadata["page_num"] for c in chunks))
        print(f"Distinct page numbers found across chunks: {page_nums_seen}")

        for idx, c in enumerate(chunks):
            print(f"\n[CHUNK {idx}] Metadata: {c.metadata}")
            print(f"Text Preview: {c.text[:]}...")