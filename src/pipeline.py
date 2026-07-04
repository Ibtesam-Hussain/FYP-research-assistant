# src/pipeline.py
"""
pipeline.py — the single entry point that wires retrieval together:

    query
      -> DenseRetriever.retrieve(top_k=N)   ──┐
      -> SparseRetriever.retrieve(top_k=N)  ──┴-> fusion.reciprocal_rank_fusion()
                                                    -> top candidates
                                                    -> Reranker.rerank(top_k=FINAL_K)
                                                    -> final ranked chunks

                                                    
IMPORTANT RAG SYSTEM DESIGN CONCEPT -------------
WHY THIS IS A CLASS, NOT A FUNCTION:
Both the embedding model (inside DenseRetriever) and the cross-encoder
(inside Reranker) are expensive to load from disk. If query() were a bare
function that built fresh DenseRetriever()/Reranker() objects every call,
every single query would reload both models from scratch -- seconds of
dead time per query for no benefit.

Instead, RAGPipeline.__init__ loads everything ONCE. Create one
RAGPipeline instance at app startup (or cache it -- see app/streamlit_app.py
using @st.cache_resource) and call .retrieve() repeatedly on that same
instance. This is the "load once, reuse" pattern that matters most for
query-time latency.
"""

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker
from src.generation.prompt_templates import build_rag_prompt
from src.generation.llm_client import LLMClient

# How many candidates each retriever pulls BEFORE fusion. Wider than the
# final answer set so fusion/reranking have real material to re-sort,
# not just re-confirm an already-narrow list.
RETRIEVAL_TOP_K = 15

# How many fused candidates get passed INTO the reranker. Capping this
# (rather than reranking everything fusion returns) keeps cross-encoder
# cost bounded -- reranking is the most expensive step per query.
RERANK_CANDIDATE_POOL = 15

# Final number of chunks returned after reranking -- this is what
# actually gets handed to the LLM for generation.
FINAL_TOP_K = 5


class RAGPipeline:
    def __init__(
        self,
        dense_retriever: DenseRetriever = None,
        sparse_retriever: SparseRetriever = None,
        reranker: Reranker = None,
        llm_client: LLMClient = None,
    ):
        # Each of these loads a model on construction (BGE embedder,
        # BM25 index, cross-encoder) -- exactly why this happens once,
        # here, in __init__, not inside retrieve().
        print("Initializing RAG pipeline (loading models)...")
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_retriever = sparse_retriever or SparseRetriever()
        self.reranker = reranker or Reranker()
        self.llm_client = llm_client or LLMClient()
        print("RAG pipeline ready.")

    def retrieve(
        self,
        query: str,
        retrieval_top_k: int = RETRIEVAL_TOP_K,
        rerank_pool: int = RERANK_CANDIDATE_POOL,
        final_top_k: int = FINAL_TOP_K,
    ) -> dict:
        """
        Runs the full hybrid retrieval pipeline for a single query.

        Returns a dict with the final chunks AND the intermediate stage
        outputs -- keeping intermediate results around is deliberate:
        eval/ will need to compare dense-only vs sparse-only vs hybrid
        configs against the SAME query, so having each stage's output
        visible (not just the final answer) saves you from re-running
        retrieval three separate times during evaluation.
        """
        dense_results = self.dense_retriever.retrieve(query, top_k=retrieval_top_k)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=retrieval_top_k)

        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        reranked = self.reranker.rerank(query, fused[:rerank_pool], top_k=final_top_k)

        return {
            "query": query,
            "dense_results": dense_results,
            "sparse_results": sparse_results,
            "fused_results": fused,
            "final_results": reranked,
        }

    def query(
        self,
        user_query: str,
        retrieval_top_k: int = RETRIEVAL_TOP_K,
        rerank_pool: int = RERANK_CANDIDATE_POOL,
        final_top_k: int = FINAL_TOP_K,
    ) -> dict:
        """
        Full pipeline: retrieve -> build prompt -> generate answer.

        Returns a dict with the final chunks AND intermediate stage
        outputs (same shape as retrieve()) plus the generated answer,
        so callers get both the answer and full retrieval provenance.
        """
        retrieval_output = self.retrieve(
            user_query,
            retrieval_top_k=retrieval_top_k,
            rerank_pool=rerank_pool,
            final_top_k=final_top_k,
        )

        final_chunks = retrieval_output["final_results"]
        messages = build_rag_prompt(user_query, final_chunks)
        answer = self.llm_client.generate(messages)

        return {
            "query": user_query,
            "answer": answer,
            "final_results": final_chunks,
            "dense_results": retrieval_output["dense_results"],
            "sparse_results": retrieval_output["sparse_results"],
            "fused_results": retrieval_output["fused_results"],
        }


if __name__ == "__main__":
    # End-to-end smoke test against your REAL index (no fake data here --
    # this is the actual integration point, so it should run on real chunks).
    pipeline = RAGPipeline()

    # test_query1 = "How does monocular depth estimation compare to stereo depth estimation?"
    # test_query = "What method does the monocular depth estimation use to handle occlusion in depth estimation?" 
    test_query = "What is the main limitation of monocular depth estimation?"
    result = pipeline.query(test_query)

    print(f"\nQuery: {result['query']}")
    print(f"\nDense candidates: {len(result['dense_results'])}")
    print(f"Sparse candidates: {len(result['sparse_results'])}")
    print(f"Fused candidates: {len(result['fused_results'])}")
    print(f"\nFinal top {len(result['final_results'])} chunks after reranking:")
    for i, r in enumerate(result["final_results"], start=1):
        print(f"\n[{i}] chunk_id={r['chunk_id']}  rerank_score={r['rerank_score']:.4f}")
        print(f"    source={r['metadata'].get('source_file')}  page={r['metadata'].get('page_num')}")
        print(f"    text preview: {r['text'][:]}")

    print(f"\n{'='*60}")
    print("ANSWER:")
    print('='*60)
    print(result["answer"])