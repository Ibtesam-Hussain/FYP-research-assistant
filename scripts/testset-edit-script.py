# scripts/update_testset_chunk_ids.py
"""
Interactive script to verify and fix relevant_chunk_ids in qa_testset.json.

For each answerable question:
1. Shows the current relevant_chunk_ids in your testset
2. Runs actual retrieval and shows top-5 chunks with text previews
3. Asks you to select which retrieved chunks actually answer the question
4. Updates the testset JSON with your selections

Run from project root:
    python scripts/update_testset_chunk_ids.py

A backup of your original testset is saved before any changes are made.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker

TESTSET_PATH = Path("data/eval/qa_testset.json")
BACKUP_DIR   = Path("data/eval/backups")


# ── lightweight retrieval (no LLM needed) ────────────────────────────────────

def retrieve(query: str, dense, sparse, reranker, top_k: int = 5) -> list[dict]:
    dense_results  = dense.retrieve(query, top_k=top_k)
    sparse_results = sparse.retrieve(query, top_k=top_k)
    fused          = reciprocal_rank_fusion(dense_results, sparse_results)
    reranked       = reranker.rerank(query, fused[:15], top_k=top_k)
    return reranked


# ── display helpers ───────────────────────────────────────────────────────────

def print_divider(char="=", width=70):
    print(char * width)

def print_chunk(rank: int, chunk: dict):
    print(f"\n  [{rank}] chunk_id : {chunk['chunk_id']}")
    print(f"       source   : {chunk['metadata'].get('source_file', '?')}")
    print(f"       page     : {chunk['metadata'].get('page_num', '?')}")
    print(f"       section  : {chunk['metadata'].get('section_heading', '?')}")
    preview = chunk["text"][:300].replace("\n", " ").strip()
    print(f"       preview  : {preview}...")


def get_user_selection(n_chunks: int) -> list[int]:
    """
    Ask user which chunk numbers contain the correct answer.
    Returns list of 0-based indices into the retrieved chunks list.
    Accepts: "1", "1 3", "1,3", "none", "s" (skip), "q" (quit)
    """
    while True:
        raw = input(
            f"\n  Select chunk numbers that answer this question "
            f"(1-{n_chunks}, space/comma separated).\n"
            f"  Or: 'none' = no chunk answers it | "
            f"'s' = skip (keep current) | 'q' = quit+save\n"
            f"  > "
        ).strip().lower()

        if raw in ("q", "quit"):
            return "quit"
        if raw in ("s", "skip"):
            return "skip"
        if raw in ("none", "n", "0"):
            return []

        # Parse numbers
        raw = raw.replace(",", " ")
        parts = raw.split()
        try:
            indices = [int(p) - 1 for p in parts]  # convert to 0-based
            if all(0 <= i < n_chunks for i in indices):
                return indices
            else:
                print(f"  ⚠ Please enter numbers between 1 and {n_chunks}.")
        except ValueError:
            print("  ⚠ Invalid input. Enter numbers like: 1  or  1 3  or  none")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # ── load testset ──────────────────────────────────────────────────────────
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # ── backup before touching anything ──────────────────────────────────────
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"qa_testset_backup_{timestamp}.json"
    shutil.copy(TESTSET_PATH, backup_path)
    print(f"Backup saved -> {backup_path}\n")

    # ── load retrieval components once ────────────────────────────────────────
    print("Loading retrieval models (once)...")
    dense   = DenseRetriever()
    sparse  = SparseRetriever()
    reranker = Reranker()
    print("Models loaded.\n")

    # ── filter to answerable questions only ───────────────────────────────────
    answerable = [q for q in questions if q.get("answerable", True)
                  and q.get("relevant_chunk_ids") is not None]
    unanswerable_ids = {q["question_id"] for q in questions
                        if not q.get("answerable", True)}

    print(f"Total questions   : {len(questions)}")
    print(f"Answerable        : {len(answerable)}")
    print(f"Unanswerable (skipped) : {len(unanswerable_ids)}")
    print(f"\nFor each question, review the top-5 retrieved chunks and")
    print(f"select which ones actually contain the correct answer.\n")
    print_divider()

    # ── build a lookup for fast update ───────────────────────────────────────
    questions_by_id = {q["question_id"]: q for q in questions}
    changed = 0

    # ── process each question ─────────────────────────────────────────────────
    for idx, item in enumerate(answerable, start=1):
        qid      = item["question_id"]
        question = item["question"]
        current_ids = item.get("relevant_chunk_ids", [])

        print(f"\n[{idx}/{len(answerable)}] {qid}")
        print(f"Q: {question}")
        print(f"Current relevant_chunk_ids: {current_ids}")

        # Retrieve
        try:
            results = retrieve(question, dense, sparse, reranker, top_k=5)
        except Exception as e:
            print(f"  ⚠ Retrieval failed: {e} — skipping.")
            continue

        print(f"\nTop-5 retrieved chunks:")
        for rank, chunk in enumerate(results, start=1):
            print_chunk(rank, chunk)

        # Get user input
        selection = get_user_selection(len(results))

        if selection == "quit":
            print("\nQuitting — saving progress so far.")
            break
        elif selection == "skip":
            print(f"  ↷ Skipped — keeping current IDs: {current_ids}")
            continue
        else:
            new_ids = [results[i]["chunk_id"] for i in selection]

            if new_ids != current_ids:
                questions_by_id[qid]["relevant_chunk_ids"] = new_ids
                changed += 1
                print(f"  ✓ Updated: {new_ids}")
            else:
                print(f"  ↷ No change.")

        print_divider("-", 70)

    # ── save updated testset ──────────────────────────────────────────────────
    updated_questions = list(questions_by_id.values())
    with open(TESTSET_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_questions, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"Done. {changed} question(s) updated.")
    print(f"Testset saved -> {TESTSET_PATH}")
    print(f"Original backup -> {backup_path}")


if __name__ == "__main__":
    main()