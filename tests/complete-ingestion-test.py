# test_retrieval.py

import numpy as np
from pathlib import Path
from src.ingestion.parser import parse_pdf
from src.ingestion.chunker import ProductionLangChainChunker
from src.ingestion.embedder import Embedder

def run_local_rag_test():
    
    pdf_path = Path("data/raw/1812.11671v1.pdf")
    if not pdf_path.exists():
        print(f"Error: Please place your test paper at: {pdf_path}")
        return

    print("--- [STAGE 1] PARSING AND CHUNKING ---")
    parsed_doc = parse_pdf(pdf_path)
    
    chunker = ProductionLangChainChunker(chunk_size_chars=1200, chunk_overlap_chars=200)
    chunks = chunker.chunk_document(parsed_doc)
    print(f"Generated {len(chunks)} chunks from the PDF.")

    print("\n--- [STAGE 2] GENERATING DENSE VECTORS ---")
    embedder = Embedder()
    # Ingestion phase: Turn plain documents into raw numeric matrix grid arrays
    document_vectors = embedder.embed_documents(chunks)
    
    print("\nINSPECTING THE VECTOR MATRIX (LAYMAN SANITY CHECK)")
    print(f"• Vector Matrix Shape: {document_vectors.shape}")
    print(f"  └─ {document_vectors.shape[0]} chunks total")
    print(f"  └─ {document_vectors.shape[1]} dimensional coordinates per chunk")
    print(f"• Sample Vector Content (First 5 dimensions of Chunk 0):")
    print(f"  {document_vectors[0][:5]} ...")

    print("\n--- [STAGE 3] SIMULATING A USER QUERY ---")

    user_query = "What architecture or training strategy did the authors propose?"
    print(f"User Query: '{user_query}'")
    
    # Retrieval phase: Embed query with the asymmetric instruction prefix warped inside
    query_vector = embedder.embed_query(user_query)
    
    print("\n--- [STAGE 4] RUNNING MATHEMATICAL RETRIEVAL (DOT PRODUCT) ---")
    # Because embedder.py normalizes vectors, Dot Product = Cosine Similarity.
    # This multiplies the query vector across all document vectors to find match scores.
    similarity_scores = np.dot(document_vectors, query_vector)
    
    # Get the indices of the top 3 highest scoring text chunks
    top_k_indices = np.argsort(similarity_scores)[::-1][:3]

    print("\nTOP 3 MOST RELEVANT CHUNKS MATCHED IN VECTOR SPACE:")
    print("=" * 80)
    for rank, idx in enumerate(top_k_indices, start=1):
        score = similarity_scores[idx]
        matched_chunk = chunks[idx]
        
        print(f"\n[RANK {rank}] Similarity Match Score: {score:.4f}")
        print(f"Metadata Profile: {matched_chunk.metadata}")
        print(f"Text Snippet:\n{matched_chunk.text[:]}")
        print("-" * 80)

if __name__ == "__main__":
    run_local_rag_test()
