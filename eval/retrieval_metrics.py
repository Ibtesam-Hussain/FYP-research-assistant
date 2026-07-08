# eval/retrieval_metrics.py
"""
retrieval_metrics.py — Recall@k and MRR across three retrieval configs.

Runs the same 28 questions through:
  - Config A: dense only   (baseline)
  - Config B: sparse only  (baseline)
  - Config C: hybrid       (your system: RRF + rerank)

Outputs a comparison table + saves full results to
eval/results/retrieval_results.json for use by run_eval.py later.

WHY THREE CONFIGS:
This is the core eval claim of your project — that hybrid retrieval
outperforms either method alone. Without this comparison, you have
"I built hybrid RAG." With it, you have "I can prove hybrid RAG
improves Recall@5 by X points over dense-only and Y points over
sparse-only on my corpus." The second sentence is what gets you
through technical interviews.
"""

import json
import time
from pathlib import Path

from src.pipeline import RAGPipeline


TESTSET_PATH  = Path("data/eval/qa_testset.json")
RESULTS_DIR   = Path("eval/results")
OUTPUT_PATH   = RESULTS_DIR / "retrieval_results.json"

# ── eval knobs ────────────────────────────────────────────────────────────────
K_VALUES = [1, 3, 5]   # compute Recall@1, Recall@3, Recall@5 simultaneously


# ─────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    1.0 if ANY relevant chunk appears in the top-k retrieved results.
    0.0 otherwise.

    Binary (not graded): we care whether the right chunk is
    retrievable at all in the top-k window, not how many relevant
    chunks appear. For a single-answer QA corpus like yours, this
    is the right formulation.
    """
    if not relevant_ids:
        return 0.0  # unanswerable questions skipped at call site
    top_k_ids = [r["chunk_id"] for r in retrieved_ids[:k]]
    return 1.0 if any(rid in top_k_ids for rid in relevant_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    1/rank of the first relevant chunk in the retrieved list.
    0.0 if no relevant chunk appears anywhere in the list.

    MRR rewards systems that surface the right chunk at rank 1
    more than rank 5 -- a more sensitive signal than Recall@k alone.
    Example: relevant chunk at rank 1 → 1.0, rank 2 → 0.5, rank 5 → 0.2
    """
    if not relevant_ids:
        return 0.0
    for rank, result in enumerate(retrieved_ids, start=1):
        if result["chunk_id"] in relevant_ids:
            return 1.0 / rank
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-config retrieval runner
# ─────────────────────────────────────────────────────────────────────────────

def run_config(
    pipeline: RAGPipeline,
    questions: list[dict],
    config_name: str,
) -> dict:
    """
    Runs all answerable questions through one retrieval config.
    Returns per-question results + aggregate metrics.

    config_name controls which part of pipeline.retrieve()'s output
    we evaluate against:
      "dense_only"  → dense_results  (embedding similarity only)
      "sparse_only" → sparse_results (BM25 only)
      "hybrid"      → final_results  (RRF + rerank)
    """
    config_key_map = {
        "dense_only":  "dense_results",
        "sparse_only": "sparse_results",
        "hybrid":      "final_results",
    }
    result_key = config_key_map[config_name]

    per_question = []
    skipped = 0

    for i, item in enumerate(questions):
        # Skip unanswerable questions for retrieval metrics --
        # they have no relevant_chunk_ids so Recall/MRR are undefined.
        if not item.get("answerable", True) or not item.get("relevant_chunk_ids"):
            skipped += 1
            continue

        query    = item["question"]
        relevant = item["relevant_chunk_ids"]

        print(f"  [{config_name}] Q{i+1:02d}/{len(questions)}: {query[:60]}...")

        retrieval_output = pipeline.retrieve(query)
        retrieved        = retrieval_output[result_key]

        recalls = {k: recall_at_k(retrieved, relevant, k) for k in K_VALUES}
        rr      = reciprocal_rank(retrieved, relevant)

        per_question.append({
            "question_id":    item["question_id"],
            "question":       query,
            "relevant_ids":   relevant,
            "retrieved_ids":  [r["chunk_id"] for r in retrieved],
            "recalls":        recalls,
            "reciprocal_rank": rr,
        })

        # Small sleep to avoid hammering embedding model back-to-back
        time.sleep(0.1)

    # ── aggregate ─────────────────────────────────────────────────────────────
    n = len(per_question)
    aggregate = {}
    for k in K_VALUES:
        aggregate[f"Recall@{k}"] = (
            round(sum(q["recalls"][k] for q in per_question) / n, 4) if n else 0.0
        )
    aggregate["MRR"] = (
        round(sum(q["reciprocal_rank"] for q in per_question) / n, 4) if n else 0.0
    )
    aggregate["n_evaluated"] = n
    aggregate["n_skipped"]   = skipped

    return {
        "config":       config_name,
        "aggregate":    aggregate,
        "per_question": per_question,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_retrieval_eval(testset_path: Path = TESTSET_PATH) -> dict:
    """
    Loads testset, runs all three configs, saves results, prints table.
    Returns the full results dict (used by run_eval.py in Phase 4).
    """
    # ── load testset ──────────────────────────────────────────────────────────
    with open(testset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {testset_path}")

    answerable = [q for q in questions if q.get("answerable", True)]
    print(f"  Answerable: {len(answerable)}  |  "
          f"Unanswerable (skipped for retrieval metrics): "
          f"{len(questions) - len(answerable)}\n")

    # ── init pipeline once, reuse across all three configs ────────────────────
    # This is the "load once, reuse" pattern -- all three configs use the
    # same pipeline instance so models aren't reloaded between configs.
    pipeline = RAGPipeline()

    # ── run configs ───────────────────────────────────────────────────────────
    results = {}
    for config in ["dense_only", "sparse_only", "hybrid"]:
        print(f"\n{'='*60}")
        print(f"Running config: {config.upper()}")
        print(f"{'='*60}")
        results[config] = run_config(pipeline, questions, config)

    # ── save full results ─────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved -> {OUTPUT_PATH}")

    return results


def print_comparison_table(results: dict) -> None:
    """Prints the comparison table you'll put in your README."""
    print(f"\n{'='*60}")
    print("RETRIEVAL EVALUATION — COMPARISON TABLE")
    print(f"{'='*60}")

    header = f"{'Config':<15}" + "".join(f"{'Recall@'+str(k):<14}" for k in K_VALUES) + f"{'MRR':<10}"
    print(header)
    print("-" * len(header))

    config_labels = {
        "dense_only":  "Dense only",
        "sparse_only": "Sparse only",
        "hybrid":      "Hybrid (ours)",
    }

    for config_name, label in config_labels.items():
        agg = results[config_name]["aggregate"]
        row = f"{label:<15}"
        for k in K_VALUES:
            row += f"{agg[f'Recall@{k}']:<14.4f}"
        row += f"{agg['MRR']:<10.4f}"
        print(row)

    print(f"\nEvaluated on {results['hybrid']['aggregate']['n_evaluated']} "
          f"answerable questions.")

    # ── per-question failure analysis ─────────────────────────────────────────
    # Shows which questions hybrid STILL fails on -- useful for README
    # "known limitations" section and honest interview discussion.
    print(f"\n{'='*60}")
    print("HYBRID FAILURES (questions where relevant chunk not in top-5):")
    print(f"{'='*60}")
    failures = [
        q for q in results["hybrid"]["per_question"]
        if q["recalls"][5] == 0.0
    ]
    if failures:
        for f in failures:
            print(f"  [{f['question_id']}] {f['question'][:80]}...")
    else:
        print("  None — perfect Recall@5 on all answerable questions.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_retrieval_eval()
    print_comparison_table(results)