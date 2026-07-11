"""Save RAG pipeline answers for each question in the QA testset.

This module is intended for a two-phase workflow:
1. Generate one answer per question from the RAG pipeline, saving each result
   incrementally to disk.
2. Reuse those saved responses later for evaluation without re-running the
   expensive generation step.

The saved output is a JSON file under data/eval/ and is written in a format
that is easy to reuse by the evaluation modules.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.pipeline import RAGPipeline

load_dotenv()

TESTSET_PATH = Path("data/eval/qa_testset.json")
OUTPUT_PATH = Path("data/eval/generated_responses.json")


def _load_questions(path: Path = TESTSET_PATH) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "questions" in payload:
        return payload.get("questions", [])
    return payload


def _save_results(results: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_testset": str(TESTSET_PATH),
        "questions": results,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _retry_query(pipeline: RAGPipeline, question: str, *, max_retries: int = 4, retry_delay: float = 2.0) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return pipeline.query(question)
        except Exception as exc:  # pragma: no cover - depends on network/provider
            last_error = exc
            if attempt < max_retries:
                time.sleep(retry_delay)
                continue
            raise RuntimeError(f"Failed after {max_retries} attempts: {exc}") from exc
    if last_error is not None:
        raise RuntimeError(f"Generation failed: {last_error}") from last_error
    raise RuntimeError("Generation failed for an unknown reason")


def save_generation_responses(
    testset_path: Path = TESTSET_PATH,
    output_path: Path = OUTPUT_PATH,
    delay_between_questions: float = 2.0,
    max_retries: int = 4,
) -> dict[str, Any]:
    """Generate answers for all questions sequentially and save them to disk.

    The file is written incrementally after each question so you do not lose
    progress if the run is interrupted.
    """
    questions = _load_questions(testset_path)
    existing_results = _load_existing_results(output_path)
    existing_map = {item.get("question_id"): item for item in existing_results if item.get("question_id")}

    pipeline = RAGPipeline()
    results: list[dict[str, Any]] = []

    for index, item in enumerate(questions, start=1):
        question_id = item.get("question_id")
        question_text = item.get("question", "")
        print(f"[{index:02d}/{len(questions)}] {question_text[:70]}...")

        if question_id and question_id in existing_map:
            existing_record = existing_map[question_id]
            results.append(existing_record)
            continue

        record: dict[str, Any] = {
            "question_id": question_id,
            "question": question_text,
            "reference_answer": item.get("reference_answer"),
            "relevant_chunk_ids": item.get("relevant_chunk_ids", []),
            "question_type": item.get("question_type"),
            "answerable": item.get("answerable", True),
            "status": "pending",
            "generated_answer": None,
            "retrieved_contexts": [],
            "retrieved_chunk_ids": [],
            "attempts": 0,
            "error": None,
        }

        try:
            result = _retry_query(
                pipeline,
                question_text,
                max_retries=max_retries,
                retry_delay=2.0,
            )
            record.update(
                {
                    "status": "success",
                    "generated_answer": result.get("answer"),
                    "retrieved_contexts": [chunk.get("text") for chunk in result.get("final_results", [])],
                    "retrieved_chunk_ids": [chunk.get("chunk_id") for chunk in result.get("final_results", [])],
                    "attempts": max_retries,
                    "error": None,
                }
            )
            print(f"  ✓ Answer generated ({len(str(result.get('answer', '')))} chars)")
        except Exception as exc:  # pragma: no cover - depends on provider/network
            record.update(
                {
                    "status": "failed",
                    "generated_answer": None,
                    "retrieved_contexts": [],
                    "retrieved_chunk_ids": [],
                    "attempts": max_retries,
                    "error": str(exc),
                }
            )
            print(f"  ✗ Failed: {exc}")

        results.append(record)
        _save_results(results, output_path)

        if index < len(questions):
            time.sleep(delay_between_questions)

    failed_records = [item for item in results if item.get("status") == "failed"]
    if failed_records:
        print(f"\nRetrying {len(failed_records)} failed questions...")
        for item in failed_records:
            question_id = item.get("question_id")
            question_text = item.get("question", "")
            print(f"[retry] {question_text[:70]}...")
            record = next((r for r in results if r.get("question_id") == question_id), item)
            try:
                result = _retry_query(
                    pipeline,
                    question_text,
                    max_retries=max_retries,
                    retry_delay=2.0,
                )
                record.update(
                    {
                        "status": "success",
                        "generated_answer": result.get("answer"),
                        "retrieved_contexts": [chunk.get("text") for chunk in result.get("final_results", [])],
                        "retrieved_chunk_ids": [chunk.get("chunk_id") for chunk in result.get("final_results", [])],
                        "attempts": max_retries,
                        "error": None,
                    }
                )
                print("  ✓ Retry succeeded")
            except Exception as exc:  # pragma: no cover - depends on provider/network
                record.update(
                    {
                        "status": "failed",
                        "generated_answer": None,
                        "retrieved_contexts": [],
                        "retrieved_chunk_ids": [],
                        "attempts": max_retries,
                        "error": str(exc),
                    }
                )
                print(f"  ✗ Retry failed: {exc}")

            # replace the existing record in the results list
            results = [r for r in results if r.get("question_id") != question_id] + [record]
            _save_results(results, output_path)
            time.sleep(delay_between_questions)

    _save_results(results, output_path)
    return {"output_path": str(output_path), "questions": results}


def load_saved_generation_responses(path: Path = OUTPUT_PATH) -> list[dict[str, Any]]:
    """Load the saved generation responses as a list of question records."""
    payload = _load_existing_results(path)
    if isinstance(payload, dict):
        return payload.get("questions", [])
    return payload


if __name__ == "__main__":
    save_generation_responses()
