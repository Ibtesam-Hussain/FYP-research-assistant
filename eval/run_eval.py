# eval/run_eval.py
"""
run_eval.py — master orchestrator for the complete evaluation suite.

Runs both phases in sequence and produces a single unified report:

    Phase 2: Retrieval metrics (Recall@k, MRR) across 3 configs
             dense-only | sparse-only | hybrid
    Phase 3: Generation metrics (Faithfulness, ResponseRelevancy)
             hybrid config only

Output:
    eval/results/eval_report.json   — full machine-readable results
    Terminal                        — unified comparison table for README

Usage:
    python -m eval.run_eval                  # run everything
    python -m eval.run_eval --retrieval-only # skip generation (no LLM calls)
    python -m eval.run_eval --generation-only # skip retrieval

WHY A SEPARATE ORCHESTRATOR:
retrieval_metrics.py and generation_metrics.py are intentionally kept
as standalone modules so you can re-run either phase independently
(e.g. re-run retrieval after fixing chunk IDs without re-running
expensive generation). run_eval.py just calls both and combines their
output into one unified report. Each module saves its own results file
so partial runs are always recoverable.
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

RESULTS_DIR   = Path("eval/results")
RETRIEVAL_OUT = RESULTS_DIR / "retrieval_results.json"
GENERATION_OUT = RESULTS_DIR / "generation_results.json"
FINAL_REPORT  = RESULTS_DIR / "eval_report.json"


# ─────────────────────────────────────────────────────────────────────────────
# Report printing
# ─────────────────────────────────────────────────────────────────────────────

def print_full_report(retrieval: dict | None, generation: dict | None) -> None:
    """
    Prints the unified eval table — this is the artifact that goes
    directly into your README and that you walk through in interviews.
    """
    width = 65

    print(f"\n{'='*width}")
    print(" FYP RESEARCH ASSISTANT — FULL EVALUATION REPORT")
    print(f"{'='*width}")

    # ── retrieval table ───────────────────────────────────────────────────────
    if retrieval:
        print("\n RETRIEVAL METRICS")
        print(f" {'─'*63}")
        print(f" {'Config':<18} {'Recall@1':<12} {'Recall@3':<12} {'Recall@5':<12} {'MRR':<10}")
        print(f" {'─'*63}")

        config_labels = {
            "dense_only":  "Dense only",
            "sparse_only": "Sparse only",
            "hybrid":      "Hybrid (ours)",
        }

        scores = {}
        for config_name, label in config_labels.items():
            agg  = retrieval[config_name]["aggregate"]
            r1   = agg["Recall@1"]
            r3   = agg["Recall@3"]
            r5   = agg["Recall@5"]
            mrr  = agg["MRR"]
            scores[config_name] = {"r1": r1, "r3": r3, "r5": r5, "mrr": mrr}
            marker = " ◄" if config_name == "hybrid" else ""
            print(f" {label:<18} {r1:<12.4f} {r3:<12.4f} {r5:<12.4f} {mrr:<10.4f}{marker}")

        print(f" {'─'*63}")

        h = scores["hybrid"]
        d = scores["dense_only"]
        s = scores["sparse_only"]
        print(f"\n Hybrid vs Dense only  →  "
              f"Recall@5 {h['r5'] - d['r5']:+.4f}  |  MRR {h['mrr'] - d['mrr']:+.4f}")
        print(f" Hybrid vs Sparse only →  "
              f"Recall@5 {h['r5'] - s['r5']:+.4f}  |  MRR {h['mrr'] - s['mrr']:+.4f}")

        n = retrieval["hybrid"]["aggregate"]["n_evaluated"]
        print(f"\n Evaluated on {n} answerable questions.")

    else:
        print("\n [Retrieval metrics not available — run with --retrieval-only or full run]")

    # ── generation table ──────────────────────────────────────────────────────
    if generation:
        agg = generation["aggregate"]

        print(f"\n GENERATION METRICS  (Hybrid config, RAGAS LLM-as-judge)")
        print(f" {'─'*63}")

        def bar(score: float, width: int = 20) -> str:
            filled = int(score * width)
            return "█" * filled + "░" * (width - filled)

        def interp(score: float) -> str:
            if score >= 0.85: return "Strong ✓"
            if score >= 0.70: return "Acceptable"
            if score >= 0.55: return "Weak"
            return "Poor ✗"

        f_score = agg.get("faithfulness", 0.0)

        # handle both key names — old runs saved "response_relevancy",
        # new runs save "answer_relevancy"
        r_score = agg.get("answer_relevancy") or agg.get("response_relevancy", 0.0)

        # count how many questions actually got scored vs null
        per_q      = generation.get("per_question", [])
        null_faith = [q for q in per_q if q.get("faithfulness") is None]
        null_relev = [q for q in per_q
                      if q.get("answer_relevancy") is None
                      and q.get("response_relevancy") is None]
        scored_n   = len(per_q) - len(null_faith)

        print(f" {'Metric':<22} {'Score':<8} {'Bar':<22} Status")
        print(f" {'─'*63}")
        print(f" {'Faithfulness':<22} {f_score:<8.4f} {bar(f_score):<22} {interp(f_score)}")
        print(f" {'Answer Relevancy':<22} {r_score:<8.4f} {bar(r_score):<22} {interp(r_score)}")
        print(f" {'─'*63}")
        print(f"\n Scores computed over {scored_n}/{len(per_q)} questions "
              f"({len(null_faith)} null scores excluded from aggregate).")

        if null_faith:
            print(f"\n ⚠ {len(null_faith)} question(s) returned null faithfulness "
                  f"(RAGAS eval failed — 429/timeout during scoring):")
            for q in null_faith:
                print(f"   [null] {q['question'][:70]}...")

        # low faithfulness — scored but below threshold
        low_faith = [q for q in per_q
                     if q.get("faithfulness") is not None
                     and q["faithfulness"] < 0.5]
        if low_faith:
            print(f"\n ⚠ {len(low_faith)} answer(s) scored <0.5 faithfulness "
                  f"(potential hallucination):")
            for q in low_faith:
                print(f"   [{q['faithfulness']:.2f}] {q['question'][:70]}...")
        elif not null_faith:
            print("\n ✓ No answers flagged for low faithfulness.")

    else:
        print("\n [Generation metrics not available — run without --retrieval-only]")

    print(f"\n{'='*width}")
    print(f" Report saved → {FINAL_REPORT}")
    print(f"{'='*width}\n")



def save_combined_report(retrieval: dict | None, generation: dict | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(),
        "retrieval":    retrieval,
        "generation":   generation,
    }
    with open(FINAL_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Load cached results (avoid re-running expensive phases unnecessarily)
# ─────────────────────────────────────────────────────────────────────────────

def load_cached(path: Path) -> dict | None:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None



def main():
    parser = argparse.ArgumentParser(
        description="FYP RAG — full evaluation orchestrator"
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run retrieval metrics only (no LLM generation calls)"
    )
    parser.add_argument(
        "--generation-only",
        action="store_true",
        help="Run generation metrics only (skips retrieval phase)"
    )
    parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Load existing results from disk instead of re-running. "
             "Useful for re-printing the report without re-running eval."
    )
    args = parser.parse_args()

    run_retrieval  = not args.generation_only
    run_generation = not args.retrieval_only

    retrieval_results  = None
    generation_results = None

    # ── cached mode: just reload and reprint ─────────────────────────────────
    if args.use_cached:
        print("Loading cached results...")
        retrieval_results  = load_cached(RETRIEVAL_OUT)
        generation_results = load_cached(GENERATION_OUT)

        if not retrieval_results and not generation_results:
            print("No cached results found. Run without --use-cached first.")
            sys.exit(1)

        save_combined_report(retrieval_results, generation_results)
        print_full_report(retrieval_results, generation_results)
        return

    
    if run_retrieval:
        print("=" * 65)
        print(" PHASE 2 — RETRIEVAL METRICS")
        print("=" * 65)

        # Import here (not top-level) so --generation-only skips model loading
        from eval.retrieval_metrics import run_retrieval_eval
        retrieval_results = run_retrieval_eval()

    else:
        # load cached if skipping — needed for combined report
        retrieval_results = load_cached(RETRIEVAL_OUT)
        if retrieval_results:
            print("Loaded cached retrieval results.")

    
    if run_generation:
        print("\n" + "=" * 65)
        print(" PHASE 3 — GENERATION METRICS (RAGAS)")
        print("=" * 65)

        from eval.generation_metrics import run_generation_eval
        generation_results = run_generation_eval(
            generated_responses_path=Path("data/eval/generated_responses.json")
        )

    else:
        generation_results = load_cached(GENERATION_OUT)
        if generation_results:
            print("Loaded cached generation results.")

   
    save_combined_report(retrieval_results, generation_results)
    print_full_report(retrieval_results, generation_results)


if __name__ == "__main__":
    main()