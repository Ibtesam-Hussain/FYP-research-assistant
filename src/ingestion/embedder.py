"""
embedder.py — generates dense vector embeddings for chunks.

Uses BAAI/bge-base-en-v1.5: a strong open embedding model that runs locally
(no API cost), and performs well on technical/scientific text — relevant
since the corpus mixes precise terminology (algorithm names, equations)
with conceptual prose (explanations of methodology).
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from src.ingestion.chunker import DocumentChunk
from pathlib import Path

LOCAL_MODEL_PATH = Path("models/Embedding models/bge-base-en")
# BGE models recommend prefixing queries (not documents) with this instruction
# for retrieval tasks — improves alignment between query and document vectors.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str = LOCAL_MODEL_PATH):
        print(f"Loading embedding model: {model_name} (first run downloads weights)")
        self.model = SentenceTransformer(str(LOCAL_MODEL_PATH))

    def embed_documents(self, chunks: list[DocumentChunk]) -> np.ndarray:
        """Embed chunk texts (no instruction prefix — documents are embedded plain)."""
        texts = [c.text for c in chunks]
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # so cosine similarity = dot product
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query with the retrieval instruction prefix."""
        prefixed = QUERY_INSTRUCTION + query
        return self.model.encode([prefixed], normalize_embeddings=True)[0]


if __name__ == "__main__":
    import sys
    from src.ingestion.parser import parse_all_pdfs, parse_pdf
    from src.ingestion.chunker import ProductionLangChainChunker
    from pathlib import Path
    
    sample_path = Path("data/raw/1901.09402v1 (1).pdf")

    if sample_path.exists():
        print("--- Testing embedding on 1 pdf ---")
        
        #parsing stage
        doc = parse_pdf(sample_path)
        
        #chunking stage
        chunker = ProductionLangChainChunker(chunk_size_chars=1500, chunk_overlap_chars=250)
        # chunks = []
        doc_chunks = chunker.chunk_document(doc)

        # for doc in doc_chunks:
        #     # chunk_document returns a list of chunks for a single PDF
        #     chunks.extend(doc)

        # embedding stage
        embedder = Embedder()
        vectors = embedder.embed_documents(doc_chunks)
        print(f"\nEmbedded {len(doc_chunks)} chunks -> shape {vectors.shape}")
        
    
    # raw_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    # docs = parse_all_pdfs(raw_dir)
    
    # # 2. Instantiate your class configuration
    # chunker = ProductionLangChainChunker(chunk_size_chars=1500, chunk_overlap_chars=250)

    # # 3. Process all documents into a single flat list of chunks
    # chunks = []
    # for doc in docs:
    #     # chunk_document returns a list of chunks for a single PDF
    #     doc_chunks = chunker.chunk_document(doc)
    #     chunks.extend(doc_chunks)

    # embedder = Embedder()
    # vectors = embedder.embed_documents(chunks)
    # print(f"\nEmbedded {len(chunks)} chunks -> shape {vectors.shape}")