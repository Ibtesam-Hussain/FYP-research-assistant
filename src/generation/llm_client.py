# src/generation/llm_client.py
"""
llm_client.py — thin wrapper around OpenRouter's chat completions API.

OpenRouter is OpenAI-compatible: same endpoint shape, same message format,
just a different base URL and API key. We use the openai Python SDK pointed
at OpenRouter's base URL rather than raw requests, since it gives us retry
logic, timeout handling, and streaming support for free.

INSTRUCTION FOR YOU IF YOU CLONE AND USE THIS REPO
--------------------------------------------------
Model choice: "google/gemini-flash-1.5" is a good default for RAG generation
-- fast, cheap, long context window (handles 5 chunks of ~1500 chars easily),
and reliable at following structured citation instructions. Swap to
"anthropic/claude-3.5-haiku" or "openai/gpt-4o-mini" if you prefer;
the prompt format works with any OpenAI-compatible model.
"""

import os
import time
import openai
from openai import OpenAI
from openai import APIError, RateLimitError, InternalServerError
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b:free"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.1  # low temperature = more faithful, less creative
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 2.0


class LLMClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    ):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found. "
            )

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def generate(self, messages: list[dict]) -> str:
        """
        Takes a messages list (system + user, from prompt_templates.py)
        and returns the LLM's response as a plain string.

        Temperature is set low (0.1) deliberately: RAG generation should
        be faithful and deterministic, not creative. Higher temperature
        increases the risk of the model drifting from the provided context.
        """
        delay = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content.strip()

            except (RateLimitError, InternalServerError, APIError) as e:
                status = getattr(e, "http_status", None)
                if isinstance(e, RateLimitError) or status == 429 or status == 503:
                    if attempt == self.max_retries:
                        raise RuntimeError(
                            f"LLM generation failed after {attempt} retries due to rate limiting or transient server error: {e}"
                        ) from e

                    print(f"LLM rate limit hit, retry {attempt}/{self.max_retries} after {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= self.retry_backoff
                    continue

                raise RuntimeError(f"LLM generation failed: {e}") from e

            except Exception as e:
                message = str(e)
                if "429" in message or "RateLimit" in type(e).__name__:
                    if attempt == self.max_retries:
                        raise RuntimeError(
                            f"LLM generation failed after {attempt} retries due to rate limiting: {e}"
                        ) from e

                    print(f"LLM rate limit hit, retry {attempt}/{self.max_retries} after {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= self.retry_backoff
                    continue

                raise RuntimeError(f"LLM generation failed: {e}") from e


if __name__ == "__main__":
    # Quick smoke test — requires OPENROUTER_API_KEY in your .env
    from src.generation.prompt_templates import build_rag_prompt

    fake_chunks = [
        {
            "text": "Monocular depth estimation predicts depth from a single RGB image using deep neural networks. Unlike stereo methods, it does not require two cameras.",
            "metadata": {"source_file": "survey_paper.pdf", "page_num": 3, "section_heading": "Introduction"},
        },
        {
            "text": "The main limitation of monocular depth estimation is scale ambiguity: without stereo baselines or known object sizes, absolute depth scale cannot be recovered.",
            "metadata": {"source_file": "survey_paper.pdf", "page_num": 5, "section_heading": "Limitations"},
        },
    ]

    query = "What is the main limitation of monocular depth estimation?"
    messages = build_rag_prompt(query, fake_chunks)

    client = LLMClient()
    answer = client.generate(messages)
    print(f"Query: {query}")
    print(f"\nAnswer:\n{answer}")