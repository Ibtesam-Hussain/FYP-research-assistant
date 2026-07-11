# # scripts/regenerate_failed.py
# """
# Finds failed or missing entries in generated_responses.json and
# re-runs them through the pipeline, updating the file in place.
# """

# import json
# import time
# from pathlib import Path
# from src.pipeline import RAGPipeline

# RESPONSES_PATH = Path("data/eval/generated_responses.json")
# TESTSET_PATH   = Path("data/eval/qa_testset.json")


# def load_responses(path: Path) -> list[dict]:
#     """Load responses regardless of whether the file is a bare list
#     or wrapped in a top-level dict like {"responses": [...]}."""
#     with open(path, encoding="utf-8") as f:
#         raw = json.load(f)

#     if isinstance(raw, list):
#         return raw
#     if isinstance(raw, dict):
#         # try common wrapper keys
#         for key in ("responses", "results", "data", "items", "questions"):
#             if key in raw and isinstance(raw[key], list):
#                 print(f"Detected wrapper key: '{key}'")
#                 return raw[key]
#         # if no wrapper key, treat values as the records
#         # (handles {"q001": {...}, "q005": {...}} shape)
#         records = list(raw.values())
#         if records and isinstance(records[0], dict):
#             print("Detected dict-of-records shape — converting to list")
#             return records
#         raise ValueError(
#             f"Unrecognised JSON shape. Top-level keys: {list(raw.keys())[:10]}"
#         )
#     raise ValueError(f"Expected list or dict, got {type(raw)}")


# def save_responses(path: Path, records: list[dict]) -> None:
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(records, f, indent=2, ensure_ascii=False)


# def main():
    
#     responses = load_responses(RESPONSES_PATH)
#     print(f"Loaded {len(responses)} response records.")

#     with open(TESTSET_PATH, encoding="utf-8") as f:
#         questions = {q["question_id"]: q for q in json.load(f)}

    
#     failed = [r for r in responses if r.get("status") == "failed"]
#     print(f"Found {len(failed)} failed response(s): "
#           f"{[r['question_id'] for r in failed]}")

#     if not failed:
#         print("Nothing to regenerate — all responses already succeeded.")
#         return

    
#     pipeline = RAGPipeline()

#     for record in failed:
#         qid   = record["question_id"]
#         item  = questions.get(qid)
#         if not item:
#             print(f"  ⚠ {qid} not found in testset — skipping.")
#             continue

#         query = item["question"]
#         print(f"\nRegenerating {qid}: {query[:65]}...")

#         try:
#             result = pipeline.query(query)

#             record["status"]             = "success"
#             record["generated_answer"]   = result["answer"]
#             record["retrieved_contexts"] = [r["text"] for r in result["final_results"]]
#             record["retrieved_chunk_ids"]= [r["chunk_id"] for r in result["final_results"]]
#             record.pop("error", None)
#             print(f"  ✓ Regenerated ({len(result['answer'])} chars)")

#         except Exception as e:
#             print(f"  ✗ Still failing: {e}")
#             record["error"] = str(e)

#         time.sleep(4)  # generous delay — free tier rate limits

    
#     save_responses(RESPONSES_PATH, responses)
#     print(f"\nSaved updated responses -> {RESPONSES_PATH}")

#     remaining = [r for r in responses if r.get("status") == "failed"]
#     if remaining:
#         print(f"⚠ {len(remaining)} still failed: {[r['question_id'] for r in remaining]}")
#         print("Wait a few minutes for rate limits to clear, then re-run this script.")
#     else:
#         print("✓ All responses now successful. Ready to run generation eval.")


# if __name__ == "__main__":
#     main()



# scripts/regenerate_failed.py
"""
Finds failed or missing entries in generated_responses.json and
re-runs them through the pipeline, retrying indefinitely until
every question has a successful response.
"""

import json
import time
from pathlib import Path
from src.pipeline import RAGPipeline

RESPONSES_PATH = Path("data/eval/generated_responses.json")
TESTSET_PATH   = Path("data/eval/qa_testset.json")

# seconds to wait between retries when rate limited
BASE_RETRY_DELAY = 10.0
MAX_RETRY_DELAY  = 120.0


def load_responses(path: Path) -> tuple[dict, list[dict]]:
    """Returns (raw_wrapper_dict, records_list) so we can save
    back in the original wrapper shape."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return None, raw

    if isinstance(raw, dict):
        for key in ("questions", "responses", "results", "data", "items"):
            if key in raw and isinstance(raw[key], list):
                return raw, raw[key]
        records = list(raw.values())
        if records and isinstance(records[0], dict):
            return None, records

    raise ValueError(f"Unrecognised JSON shape. Keys: {list(raw.keys())[:10]}")


def save_responses(path: Path, wrapper: dict | None, records: list[dict]) -> None:
    if wrapper is not None:
        # write back into the original wrapper structure
        for key in ("questions", "responses", "results", "data", "items"):
            if key in wrapper:
                wrapper[key] = records
                break
        with open(path, "w", encoding="utf-8") as f:
            json.dump(wrapper, f, indent=2, ensure_ascii=False)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)


def is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in [
        "429", "rate limit", "rate-limit",
        "too many requests", "retry later",
        "temporarily unavailable", "503",
    ])


def regenerate_one(pipeline: RAGPipeline, query: str, record: dict) -> bool:
    """
    Retries a single query indefinitely until it succeeds.
    Returns True when successful.
    Delay doubles on each rate-limit hit, capped at MAX_RETRY_DELAY.
    """
    attempt   = 0
    delay     = BASE_RETRY_DELAY

    while True:
        attempt += 1
        try:
            result = pipeline.query(query)
            record["status"]              = "success"
            record["generated_answer"]    = result["answer"]
            record["retrieved_contexts"]  = [r["text"] for r in result["final_results"]]
            record["retrieved_chunk_ids"] = [r["chunk_id"] for r in result["final_results"]]
            record.pop("error", None)
            record["attempts"] = attempt
            print(f"  ✓ Success on attempt {attempt} ({len(result['answer'])} chars)")
            return True

        except Exception as e:
            if is_rate_limit(e):
                print(f"  ⚠ Rate limited (attempt {attempt}) — waiting {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, MAX_RETRY_DELAY)  # exponential backoff
            else:
                # non-rate-limit error — still retry but log it
                print(f"  ✗ Error (attempt {attempt}): {e}")
                print(f"    Retrying in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 1.5, MAX_RETRY_DELAY)


def main():
    wrapper, responses = load_responses(RESPONSES_PATH)
    print(f"Loaded {len(responses)} response records.")

    with open(TESTSET_PATH, encoding="utf-8") as f:
        questions = {q["question_id"]: q for q in json.load(f)}

    # build lookup by question_id for fast update
    records_by_id = {str(r["question_id"]): r for r in responses}

    failed = [r for r in responses if r.get("status") == "failed"]
    print(f"Found {len(failed)} failed response(s): "
          f"{[r['question_id'] for r in failed]}\n")

    if not failed:
        print("Nothing to regenerate — all responses already succeeded.")
        return

    pipeline = RAGPipeline()

    for i, record in enumerate(failed, start=1):
        qid  = record["question_id"]
        item = questions.get(qid)

        if not item:
            print(f"[{i}/{len(failed)}] ⚠ {qid} not found in testset — skipping.")
            continue

        query = item["question"]
        print(f"\n[{i}/{len(failed)}] {qid}: {query[:65]}...")
        print(f"  Retrying indefinitely until success...")

        regenerate_one(pipeline, query, record)

        # save after every successful question so progress is never lost
        # if the script is interrupted mid-run
        save_responses(RESPONSES_PATH, wrapper, responses)
        print(f"  💾 Progress saved.")

        # small breathing room between questions even after success
        if i < len(failed):
            time.sleep(3)

    # ── final status ──────────────────────────────────────────────────────────
    still_failed = [r for r in responses if r.get("status") == "failed"]
    print(f"\n{'='*55}")
    if still_failed:
        print(f"⚠ {len(still_failed)} still failed (shouldn't happen with infinite retry):")
        for r in still_failed:
            print(f"  {r['question_id']}: {r.get('error', 'unknown')}")
    else:
        print(f"✓ All {len(failed)} previously-failed responses now successful.")
        print(f"  Ready to run: python -m eval.run_eval --generation-only")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()