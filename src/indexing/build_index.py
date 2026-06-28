"""Orchestrate ingestion and build both dense and sparse indexes.

This module ties together the parser, chunker, embedder and index
creators. It intentionally keeps the logic simple: parse -> chunk ->
embed -> index. Index builder functions are expected to exist in
`dense_index.py` and `sparse_index.py` (they may be stubs during early
development).
"""

from pathlib import Path
from typing import List
import argparse

from src.ingestion.parser import parse_all_pdfs
from src.ingestion.chunker import ProductionLangChainChunker, DocumentChunk
from src.ingestion.embedder import Embedder
from src.indexing.dense_index import DenseIndex
from src.indexing.sparse_index import SparseIndex


def build_indexes(source_dir: str = "data/raw", output_dir: str = "data/processed", 
                  chunk_size: int = 1500, chunk_overlap: int = 250):
    """Run parser, chunker, embedder, and index builders.

    Returns a tuple (chunks, embeddings) for downstream inspection.
    """
    source = Path(source_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Parsing PDFs in: {source}")
    parsed_docs = parse_all_pdfs(source)

    print("Chunking parsed documents (page-aware)...")
    chunker = ProductionLangChainChunker(chunk_size_chars=chunk_size, chunk_overlap_chars=chunk_overlap)
    chunks: List[DocumentChunk] = chunker.chunk_all_documents(parsed_docs)

    if not chunks:
        print("No chunks generated — aborting index build.")
        return [], None

    print(f"Embedding {len(chunks)} chunks (this may take a while)...")
    embedder = Embedder()
    embeddings = embedder.embed_documents(chunks)

    print("Creating dense index...")
    try:
        dense_index = DenseIndex()
        dense_index.upsert_chunks(chunks, embeddings)
    except Exception as e:
        print(f"Warning: dense index creation failed: {e}")

    print("Creating sparse index...")
    try:
        sparse_index = SparseIndex()
        sparse_index.build(chunks)
        sparse_index.save()
    except Exception as e:
        print(f"Warning: sparse index creation failed: {e}")

    print("Build finished.")
    return chunks, embeddings


def _cli():
    p = argparse.ArgumentParser(description="Build dense + sparse indexes from PDFs")
    p.add_argument("--raw_dir", default="data/raw", help="Directory with raw PDF files")
    p.add_argument("--out_dir", default="data/processed", help="Directory to store processed artifacts")
    p.add_argument("--chunk_size", type=int, default=1500)
    p.add_argument("--chunk_overlap", type=int, default=250)
    args = p.parse_args()

    build_indexes(args.raw_dir, args.out_dir, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    _cli()
