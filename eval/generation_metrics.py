# eval/generation_metrics.py
"""
generation_metrics.py — RAGAS-based generation quality evaluation.

Runs two metrics on your hybrid RAG pipeline's generated answers:

  Faithfulness     — are all claims in the answer supported by the
                     retrieved context? (detects hallucination)
                     Score: 0.0-1.0, higher is better.
                     Does NOT need reference_answer.

  AnswerRelevancy  — does the answer actually address the question?
                     (detects topic drift / off-topic answers)
                     Score: 0.0-1.0, higher is better.
                     Does NOT need reference_answer.

We skip Context Recall (the only RAGAS metric that needs reference_answer)
because your reference answers are approximate, and an inaccurate reference
answer would produce misleading Context Recall scores -- worse than not
measuring it at all.

WHY SEQUENTIAL SCORING:
RAGAS's default parallel async mode hammers the API with many simultaneous
calls, triggering 429s on free-tier models. run_ragas_sequential() scores
one question at a time with a configurable delay between calls, keeping
well under free-tier TPM limits at the cost of ~12-15 minutes total runtime.

WHY LLM-AS-JUDGE:
Both metrics work by making their own LLM calls to evaluate quality,
not by string matching or keyword overlap. Faithfulness decomposes your
answer into atomic claims and checks each one against the context.
AnswerRelevancy generates reverse questions from your answer and checks
if they match the original query.
"""

import json
import time
import os
import math
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from src.pipeline import RAGPipeline
from src.rag_generation_response_saver import (
    OUTPUT_PATH as SAVED_RESPONSES_PATH,
    load_saved_generation_responses,
)

load_dotenv()

# ── paths ─────────────────────────────────────────────────────────────────────
TESTSET_PATH      = Path("data/eval/qa_testset.json")
RESULTS_DIR       = Path("eval/results")
OUTPUT_PATH       = RESULTS_DIR / "generation_results.json"
RAGAS_CACHE_PATH  = RESULTS_DIR / "ragas_scores_cache.json"

# ── RAGAS judge model ─────────────────────────────────────────────────────────
# Points at Groq via an openai.OpenAI client (RAGAS requires OpenAI instance).
# Use Groq model string format, not OpenRouter format.
RAGAS_JUDGE_MODEL = "openai/gpt-oss-120b"
# RAGAS_JUDGE_MODEL = "qwen/qwen3-32b"

# One-time eval tuning — gpt-oss-120b needs higher max_tokens for faithfulness JSON.
RAGAS_MAX_TOKENS          = 4096
RAGAS_TEMPERATURE         = 0.1
RAGAS_ANSWER_REL_STRICTNESS = 2   # 2 reverse-questions per answer (3 = more API calls)
RAGAS_RETRY_MAX           = 5
RAGAS_RETRY_BASE_DELAY    = 30.0  # seconds; doubles on each 429


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS cache helpers — score once, reload forever
# ─────────────────────────────────────────────────────────────────────────────

def load_ragas_cache() -> list[dict] | None:
    """Load previously computed per-question RAGAS scores from disk."""
    if RAGAS_CACHE_PATH.exists():
        with open(RAGAS_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_ragas_cache(per_question: list[dict]) -> None:
    """Persist per-question RAGAS scores so re-runs skip all LLM calls."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAGAS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(per_question, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS import shim — handles missing langchain_community.chat_models.vertexai
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_ragas_optional_imports() -> None:
    """Stub out the optional VertexAI import that some RAGAS builds require."""
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import sys, types

    if "langchain_community" not in sys.modules:
        lc = types.ModuleType("langchain_community")
        lc.__path__ = []
        lc.__package__ = "langchain_community"
        sys.modules["langchain_community"] = lc

    if "langchain_community.chat_models" not in sys.modules:
        cm = types.ModuleType("langchain_community.chat_models")
        cm.__path__ = []
        cm.__package__ = "langchain_community.chat_models"
        sys.modules["langchain_community.chat_models"] = cm
        sys.modules["langchain_community"].chat_models = cm

    vertexai_mod = types.ModuleType("langchain_community.chat_models.vertexai")
    vertexai_mod.__package__ = "langchain_community.chat_models"

    class ChatVertexAI:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("VertexAI is not available in this environment")

    vertexai_mod.ChatVertexAI = ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_mod
    sys.modules["langchain_community.chat_models"].vertexai = vertexai_mod


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in [
        "429", "rate limit", "rate-limit",
        "too many requests", "retry later",
        "temporarily unavailable", "503",
    ])


def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return _is_rate_limit_error(exc) or any(t in msg for t in [
        "max_tokens", "incomplete",
        "timeout", "timed out",
        "connection", "503", "502",
    ])


def _score_with_retry(score_fn, label: str) -> float | None:
    """Run a RAGAS metric with exponential backoff on 429 / transient failures."""
    for attempt in range(RAGAS_RETRY_MAX):
        try:
            score = score_fn()
            return round(float(score), 4) if not math.isnan(float(score)) else None
        except Exception as e:
            if _is_retryable_error(e) and attempt < RAGAS_RETRY_MAX - 1:
                wait = RAGAS_RETRY_BASE_DELAY * (2 ** attempt)
                reason = "rate limited" if _is_rate_limit_error(e) else "transient error"
                print(f"    ⏳ {label} {reason} — retry {attempt + 1}/{RAGAS_RETRY_MAX} in {wait:.0f}s...")
                time.sleep(wait)
            else:
                print(f"    ⚠ {label} failed: {e}")
                return None
    return None


def _is_scored(record: dict) -> bool:
    """True only when both metrics succeeded (non-null)."""
    return (
        record.get("faithfulness") is not None
        and record.get("answer_relevancy") is not None
    )


# ─────────────────────────────────────────────────────────────────────────────
# RAGAS LLM + embeddings setup
# ─────────────────────────────────────────────────────────────────────────────

def build_ragas_llm():
    """
    Configure RAGAS judge using Groq via openai.OpenAI client.
    RAGAS llm_factory requires an openai.OpenAI instance — Groq SDK
    client fails its type check even though the API is compatible.
    Workaround: use openai.OpenAI pointed at Groq's base URL.
    """
    _ensure_ragas_optional_imports()
    from ragas.llms import llm_factory
    from openai import OpenAI

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")

    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=api_key,
    )
    return llm_factory(
        model=RAGAS_JUDGE_MODEL,
        provider="openai",
        client=client,
        max_tokens=RAGAS_MAX_TOKENS,
        temperature=RAGAS_TEMPERATURE,
    )


def build_ragas_embeddings():
    """
    Local BGE embeddings for legacy AnswerRelevancy (embed_query/embed_documents).
    Reuses the same model already downloaded for ingestion.
    """
    import torch
    from ragas.embeddings import BaseRagasEmbeddings
    from sentence_transformers import SentenceTransformer

    LOCAL_MODEL_PATH = Path("models/Embedding models/bge-base-en")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using embedding device: {device}")

    class BGEEmbeddings(BaseRagasEmbeddings):
        def __init__(self):
            super().__init__()
            self.model = SentenceTransformer(str(LOCAL_MODEL_PATH), device=device)

        def embed_query(self, text: str) -> list[float]:
            return self.model.encode(
                [text], normalize_embeddings=True
            )[0].tolist()

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.model.encode(
                texts, normalize_embeddings=True
            ).tolist()

        async def aembed_query(self, text: str) -> list[float]:
            return self.embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return self.embed_documents(texts)

    return BGEEmbeddings()


# ─────────────────────────────────────────────────────────────────────────────
# Sequential RAGAS scorer — one question at a time, no parallel API hammering
# ─────────────────────────────────────────────────────────────────────────────

def run_ragas_sequential(
    samples: list[dict],
    ragas_llm,
    ragas_embeddings,
    delay_between: float = 15.0,
) -> list[dict]:
    """
    Scores each sample one at a time with a delay between API calls.

    RAGAS's default evaluate() runs async parallel calls which hammer
    free-tier rate limits. This function scores sequentially, saving
    results after every question so progress is never lost on interruption.

    delay_between: seconds between each individual metric API call.
    With 2 metrics per question and 25 questions:
        15s delay → ~12 min total
        10s delay → ~8 min total (riskier for tight rate limits)
    """
    _ensure_ragas_optional_imports()
    from ragas import SingleTurnSample
    from ragas.metrics import AnswerRelevancy, Faithfulness
    from ragas.run_config import RunConfig
    import asyncio

    run_config = RunConfig(max_retries=3, max_wait=90, timeout=300)

    faithfulness_metric = Faithfulness(llm=ragas_llm)
    answer_rel_metric   = AnswerRelevancy(
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        strictness=RAGAS_ANSWER_REL_STRICTNESS,
    )
    faithfulness_metric.init(run_config)
    answer_rel_metric.init(run_config)

    # load existing cache so interrupted runs resume where they left off
    existing_cache = load_ragas_cache() or []
    already_scored = {q["question"]: q for q in existing_cache}

    per_question   = list(existing_cache)
    question_to_idx = {q["question"]: idx for idx, q in enumerate(per_question)}

    for i, s in enumerate(samples, start=1):
        question = s["user_input"]

        existing = already_scored.get(question)
        if existing and _is_scored(existing):
            print(f"  [{i}/{len(samples)}] Cached: {question[:60]}...")
            continue

        if existing:
            print(f"  [{i}/{len(samples)}] Re-scoring incomplete: {question[:60]}...")
        else:
            print(f"  Scoring [{i}/{len(samples)}]: {question[:60]}...")

        sample = SingleTurnSample(
            user_input=question,
            response=s["response"],
            retrieved_contexts=s["retrieved_contexts"],
        )

        # ── faithfulness ──────────────────────────────────────────────────────
        f_val = _score_with_retry(
            lambda: asyncio.run(faithfulness_metric.single_turn_ascore(sample)),
            "Faithfulness",
        )

        time.sleep(delay_between)

        # ── answer relevancy ──────────────────────────────────────────────────
        r_val = _score_with_retry(
            lambda: asyncio.run(answer_rel_metric.single_turn_ascore(sample)),
            "Answer relevancy",
        )

        record = {
            "question":         question,
            "answer":           s["response"],
            "faithfulness":     f_val,
            "answer_relevancy": r_val,
        }
        if question in question_to_idx:
            per_question[question_to_idx[question]] = record
        else:
            question_to_idx[question] = len(per_question)
            per_question.append(record)
        already_scored[question] = record

        # save after every question — progress survives interruption
        save_ragas_cache(per_question)
        print(f"    ✓ faithfulness={f_val}  answer_relevancy={r_val}  💾 saved")

        if i < len(samples):
            time.sleep(delay_between)

    return per_question


# ─────────────────────────────────────────────────────────────────────────────
# Main eval function
# ─────────────────────────────────────────────────────────────────────────────

def run_generation_eval(
    testset_path: Path = TESTSET_PATH,
    generated_responses_path: Path | None = SAVED_RESPONSES_PATH,
    delay_between_queries: float = 2.0,
    ragas_delay: float = 20.0,
) -> dict:
    """
    Full generation eval pipeline:
    1. Load QA testset (answerable questions only)
    2. Load cached generated answers, fall back to live pipeline if missing
    3. Check RAGAS score cache — skip scoring if already complete
    4. Run sequential RAGAS scoring with per-call delay
    5. Compute aggregates and save results
    """

    # ── load testset ──────────────────────────────────────────────────────────
    with open(testset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    answerable = [q for q in questions if q.get("answerable", True)]
    print(f"Loaded {len(questions)} questions.")
    print(f"Running generation eval on {len(answerable)} answerable questions.\n")

    # ── load cached generated answers ─────────────────────────────────────────
    records_by_id: dict[str, dict] = {}
    if generated_responses_path is not None and generated_responses_path.exists():
        saved_records = load_saved_generation_responses(generated_responses_path)
        records_by_id = {
            str(r.get("question_id")): r
            for r in saved_records
            if r.get("question_id") is not None
        }
        print(f"Loaded {len(saved_records)} cached responses from {generated_responses_path}")
    else:
        print("No cached responses found — will generate answers live.")

    # ── collect samples ───────────────────────────────────────────────────────
    pipeline = None
    samples  = []
    failed   = []

    for i, item in enumerate(answerable, start=1):
        query       = item["question"]
        question_id = item.get("question_id")
        record      = records_by_id.get(str(question_id))

        print(f"[{i:02d}/{len(answerable)}] {query[:70]}...")

        if record and record.get("status") == "success" and record.get("generated_answer"):
            samples.append({
                "user_input":         query,
                "response":           record["generated_answer"],
                "retrieved_contexts": record.get("retrieved_contexts", []),
            })
            print(f"  ✓ Cached answer ({len(str(record['generated_answer']))} chars)")

        elif record and record.get("status") == "failed":
            print(f"  ✗ Cached generation failed: {record.get('error', 'unknown')}")
            failed.append({"question_id": question_id, "error": record.get("error")})

        else:
            # lazy-init pipeline only if needed
            if pipeline is None:
                pipeline = RAGPipeline()
            try:
                result = pipeline.query(query)
                samples.append({
                    "user_input":         query,
                    "response":           result["answer"],
                    "retrieved_contexts": [r["text"] for r in result["final_results"]],
                })
                print(f"  ✓ Generated live ({len(result['answer'])} chars)")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                failed.append({"question_id": question_id, "error": str(e)})

        if i < len(answerable):
            time.sleep(delay_between_queries)

    print(f"\nCollected {len(samples)} answers. ({len(failed)} failed)")

    if not samples:
        raise RuntimeError("No answers available — check your pipeline and cache.")

    # ── check RAGAS cache first ───────────────────────────────────────────────
    cached_scores = load_ragas_cache()
    n_complete = sum(1 for q in (cached_scores or []) if _is_scored(q))
    if cached_scores and n_complete >= len(samples):
        print(f"\nLoading cached RAGAS scores ({n_complete}/{len(samples)} complete).")
        print(f"Delete {RAGAS_CACHE_PATH} to force a fresh evaluation.\n")
        per_question = cached_scores
    else:
        if cached_scores:
            remaining = len(samples) - n_complete
            print(f"\nPartial RAGAS cache found ({n_complete}/{len(samples)} complete).")
            print(f"Resuming — {remaining} questions left to score.\n")
        else:
            print(f"\nNo RAGAS cache found — scoring all {len(samples)} questions.")
            print(f"Estimated time: ~{len(samples) * ragas_delay * 2 / 60:.0f} minutes\n")

        print(f"Configuring RAGAS judge: {RAGAS_JUDGE_MODEL}")
        ragas_llm        = build_ragas_llm()
        ragas_embeddings = build_ragas_embeddings()

        per_question = run_ragas_sequential(
            samples,
            ragas_llm,
            ragas_embeddings,
            delay_between=ragas_delay,
        )

    # ── aggregate ─────────────────────────────────────────────────────────────
    scored_faith = [q["faithfulness"]     for q in per_question if q.get("faithfulness")     is not None]
    scored_relev = [q["answer_relevancy"] for q in per_question if q.get("answer_relevancy") is not None]

    faithfulness = round(float(np.mean(scored_faith)), 4) if scored_faith else 0.0
    relevancy    = round(float(np.mean(scored_relev)), 4) if scored_relev else 0.0

    null_faith = len(per_question) - len(scored_faith)
    null_relev = len(per_question) - len(scored_relev)

    if null_faith > 0:
        print(f"\n⚠ {null_faith} faithfulness scores are null (API failures during scoring).")
    if null_relev > 0:
        print(f"⚠ {null_relev} answer_relevancy scores are null.")

    output = {
        "aggregate": {
            "faithfulness":     faithfulness,
            "answer_relevancy": relevancy,
            "n_evaluated":      len(samples),
            "n_scored":         len(scored_faith),
            "n_null":           null_faith,
            "n_failed":         len(failed),
        },
        "per_question": per_question,
        "failed":       failed,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved -> {OUTPUT_PATH}")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Standalone table printer
# ─────────────────────────────────────────────────────────────────────────────

def print_generation_table(results: dict) -> None:
    print(f"\n{'='*58}")
    print("GENERATION QUALITY — RAGAS METRICS (Hybrid config)")
    print(f"{'='*58}")

    agg = results["aggregate"]

    def interpret(score: float) -> str:
        if score >= 0.85: return "Strong"
        if score >= 0.70: return "Acceptable"
        if score >= 0.55: return "Weak"
        return "Poor"

    f_score = agg.get("faithfulness", 0.0)
    r_score = agg.get("answer_relevancy") or agg.get("response_relevancy", 0.0)

    n_scored = agg.get("n_scored", agg.get("n_evaluated", "?"))
    n_total  = agg.get("n_evaluated", "?")
    n_null   = agg.get("n_null", 0)

    print(f"{'Metric':<25} {'Score':<10} {'Interpretation'}")
    print("-" * 58)
    print(f"{'Faithfulness':<25} {f_score:<10.4f} {interpret(f_score)}")
    print(f"{'Answer Relevancy':<25} {r_score:<10.4f} {interpret(r_score)}")
    print(f"\nScores computed over {n_scored}/{n_total} questions "
          f"({n_null} null scores excluded).")
    print(f"Generation failures: {agg.get('n_failed', 0)}")

    per_q = results.get("per_question", [])

    low_faith = [q for q in per_q
                 if q.get("faithfulness") is not None and q["faithfulness"] < 0.5]
    null_faith = [q for q in per_q if q.get("faithfulness") is None]

    if low_faith:
        print(f"\n{'='*58}")
        print("LOW FAITHFULNESS (< 0.5 — potential hallucination):")
        print(f"{'='*58}")
        for q in low_faith:
            print(f"  [{q['faithfulness']:.2f}] {q['question'][:80]}...")

    if null_faith:
        print(f"\n{'='*58}")
        print(f"NULL FAITHFULNESS ({len(null_faith)} questions — API failures during scoring):")
        print(f"{'='*58}")
        for q in null_faith:
            print(f"  [null] {q['question'][:80]}...")

    if not low_faith and not null_faith:
        print("\n✓ All questions scored with no low-faithfulness answers.")


if __name__ == "__main__":
    results = run_generation_eval()
    print_generation_table(results)