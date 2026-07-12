<div align="center">

# FYP Research Assistant

> A production-grade inspired Hybrid RAG (Retrieval-Augmented Generation) system
> for querying academic research papers on depth estimation (My FYP) — built as a
> portfolio project demonstrating end-to-end AI engineering from ingestion
> through evaluation.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fyp-research-assistant-a4gb5kde2p24dphfc8pgtz.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![RAGAS](https://img.shields.io/badge/Eval-RAGAS-8A2BE2)](https://docs.ragas.io)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-ff6b35)](https://openrouter.ai)
[![Groq](https://img.shields.io/badge/Judge-Groq-f55036)](https://groq.com)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-1a1a2e)](https://www.trychroma.com)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Pipeline Deep Dive](#pipeline-deep-dive)
  - [Stage 1 — Ingestion](#stage-1--ingestion)
  - [Stage 2 — Indexing](#stage-2--indexing)
  - [Stage 3 — Retrieval](#stage-3--retrieval)
  - [Stage 4 — Generation](#stage-4--generation)
- [Evaluation](#evaluation)
- [Models & Technology Stack](#models--technology-stack)
- [Key Design Decisions](#key-design-decisions)
- [Known Limitations](#known-limitations)
- [Running Locally](#running-locally)
- [Deployment](#deployment)
- [Results](#results)

---

## Overview

The FYP Research Assistant answers technical questions grounded in a
corpus of 8 academic papers on monocular and stereo depth estimation.
Every answer is:

- **Grounded** — generated only from retrieved paper chunks, never from
  the model's training knowledge
- **Cited** — inline `[N]` citations with source file and page number
- **Honest** — explicitly says "not found in context" when the corpus
  doesn't support an answer

The system demonstrates a complete production-oriented RAG engineering
workflow: structured ingestion, dual-index hybrid retrieval, cross-encoder
reranking, citation-enforcing generation, quantitative evaluation across
three retrieval configurations, and a deployed Streamlit UI.

---

## Architecture
![Architecture](arch1.png)

---

## Project Structure
```
FYP-research-assistant/
│
├── data/
│   ├── raw/                        # source PDFs (8 papers)
│   ├── chroma_db/                  # persistent ChromaDB vector index
│   ├── bm25_index.pkl              # serialized BM25 sparse index
│   └── eval/
│       ├── qa_testset.json         # 28 hand-authored Q&A pairs
│       └── generated_responses.json # cached LLM answers for eval
│
├── src/
│   ├── ingestion/
│   │   ├── parser.py               # PDF → structured markdown (pymupdf4llm)
│   │   ├── chunker.py              # structure-aware LangChain chunking
│   │   └── embedder.py             # BGE embedding generation
│   ├── indexing/
│   │   ├── dense_index.py          # ChromaDB wrapper (upsert + query)
│   │   ├── sparse_index.py         # BM25 wrapper (build + save + load)
│   │   └── build_index.py          # one-shot ingestion orchestrator
│   ├── retrieval/
│   │   ├── dense_retriever.py      # embed query → ChromaDB search
│   │   ├── sparse_retriever.py     # tokenize query → BM25 search
│   │   ├── fusion.py               # Reciprocal Rank Fusion (RRF)
│   │   └── reranker.py             # cross-encoder reranking
│   ├── generation/
│   │   ├── prompt_templates.py     # citation-enforcing system prompt
│   │   └── llm_client.py           # OpenRouter API wrapper + retry
│   └── pipeline.py                 # RAGPipeline class (retrieve + query)
│
├── eval/
│   ├── retrieval_metrics.py        # Recall@k + MRR across 3 configs
│   ├── generation_metrics.py       # RAGAS Faithfulness + AnswerRelevancy
│   ├── run_eval.py                 # master orchestrator (--retrieval-only etc)
│   └── results/
│       ├── retrieval_results.json
│       ├── generation_results.json
│       ├── ragas_scores_cache.json
│       └── eval_report.json
│
├── app/
│   ├── streamlit_app.py            # main entry point (thin orchestrator)
│   ├── components/
│   │   ├── sidebar.py              # settings + corpus stats + examples
│   │   ├── chat.py                 # session state + history rendering
│   │   ├── sources.py              # chunk display + retrieval debug
│   │   └── query_handler.py        # pipeline execution + streaming
│   └── utils/
│       ├── loaders.py              # @st.cache_resource model loading
│       └── helpers.py              # constants + api key + rate limit check
│
├── scripts/
│   ├── build_index.sh              # one-shot ingestion runner
│   ├── run_eval.sh                 # full eval runner
│   ├── update_testset_chunk_ids.py # interactive testset verifier
│   ├── regenerate_failed.py        # retry failed LLM responses
│   └── generate_reference_answers.py
│
├── .streamlit/
│   └── config.toml                 # headless + CORS config for deployment
├── requirements.txt
└── README.md
```
---

## Pipeline Deep Dive

### Stage 1 — Ingestion

**Goal:** Convert raw PDFs into clean, metadata-tagged text chunks ready
for indexing.

#### 1.1 Parsing — `src/ingestion/parser.py`

Uses `pymupdf4llm.to_markdown()` with `page_chunks=True` to convert each
PDF page into structured Markdown. This preserves section headings as
`#`/`##` syntax, handles two-column academic paper layouts correctly, and
attaches page numbers via `page_dict["metadata"]["page_number"]`.

**Key design decision:** We use `pymupdf4llm` rather than a raw
`PyMuPDF` font-size heuristic because research papers have highly
variable formatting. The font-size approach (tried first) failed to
distinguish section headers from page numbers and arXiv watermarks —
`pymupdf4llm`'s native structural engine handles these correctly.

Output: `ParsedDocument` objects with per-page `PageContent` (markdown
text + page number).

#### 1.2 Chunking — `src/ingestion/chunker.py`

Uses LangChain's `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`
in a two-phase approach:

**Phase 1 — Structural split:** `MarkdownHeaderTextSplitter` splits each
page's markdown on `#`, `##`, `###` headers, attaching `Header_1`,
`Header_2`, `Header_3` metadata to each section. Headers are preserved
inside chunk text (`strip_headers=False`) so the LLM sees section context.

**Phase 2 — Size refinement:** `RecursiveCharacterTextSplitter` with
`chunk_size=1500 chars`, `chunk_overlap=250 chars` breaks oversized
sections into overlapping chunks using separator hierarchy
`["\n\n", "\n", " ", ""]` — never cutting mid-sentence.

**Critical design decision:** We chunk **page-by-page**, not by merging
all pages into one global stream. The merged-stream approach (tried first)
caused a silent bug: `MarkdownHeaderTextSplitter` rewrites whitespace
(`"\n\n"` → `"  \n"`) so chunk text never byte-matches the original
stream. `str.find()` returned `-1` on every chunk, silently defaulting
every `page_num` to `1`. Per-page chunking sidesteps this entirely —
`page.page_num` is always correct by construction.

A `last_known_heading` variable tracks the most recent section heading
across page boundaries, so chunks that start mid-section inherit the
correct heading rather than a generic fallback.

Output: `DocumentChunk` objects with `text`, `source_file`, `page_num`,
`section_heading`, `chunk_id`.

#### 1.3 Embedding — `src/ingestion/embedder.py`

Uses `BAAI/bge-base-en-v1.5` via `sentence-transformers`:

- **Documents** embedded without instruction prefix, batch size 32,
  `normalize_embeddings=True` (so cosine similarity = dot product)
- **Queries** embedded with BGE's recommended retrieval instruction prefix:
  `"Represent this sentence for searching relevant passages: "`

The query prefix matters: BGE models are trained with this asymmetric
scheme specifically for retrieval tasks. Skipping it measurably degrades
retrieval quality on technical text.

#### 1.4 Orchestration — `src/indexing/build_index.py`

Runs the full parse → chunk → embed → index pipeline in one command:

```bash
python -m src.indexing.build_index          # first run
python -m src.indexing.build_index --force  # force rebuild
```

Saves `data/processed/chunks.json` as a checkpoint after chunking —
if only the embedding model changes later, chunking is skipped and
embedding runs directly from the checkpoint.

---

### Stage 2 — Indexing

Two separate indexes are built over the same 611 chunks, serving
complementary retrieval functions.

#### 2.1 Dense Index — `src/indexing/dense_index.py`

Wraps `chromadb.PersistentClient` pointed at `data/chroma_db/`. Uses
`collection.upsert()` (not `add()`) so re-running ingestion after adding
new papers overwrites existing entries rather than erroring on duplicates.

Metadata sanitization strips `None` values before upsert — ChromaDB
rejects `None` metadata fields silently in some versions, causing
retrieval misses.

**Distance direction:** ChromaDB returns L2 distance where **lower =
more similar**. This is opposite to BM25 where **higher = more relevant**.
The RRF fusion step (Stage 3) sidesteps this mismatch by using only rank
position, never raw scores.

#### 2.2 Sparse Index — `src/indexing/sparse_index.py`

Wraps `rank_bm25.BM25Okapi` with a simple whitespace+lowercase tokenizer.
BM25 has no built-in persistence, so the index is serialized with `pickle`
to `data/bm25_index.pkl` after building and loaded on startup.

**Why BM25 alongside dense:** Technical research papers mix precise
terminology (algorithm names: "MAGSAC", "SGBM", "WLS"; dataset names:
"KITTI") with conceptual prose. Dense embeddings excel at semantic
similarity but can blur together a specific named algorithm with other
related terms. BM25's exact-match scoring catches these precise technical
terms that embeddings might miss.

---

### Stage 3 — Retrieval

The retrieval layer runs on every query. All components are loaded once at
startup via `@st.cache_resource` (Streamlit) or `RAGPipeline.__init__`
(API usage) — never reloaded per query.

#### 3.1 Dual Retrieval

```python
dense_results  = dense_retriever.retrieve(query, top_k=15)
sparse_results = sparse_retriever.retrieve(query, top_k=15)
```

Both run independently and in full. `top_k=15` is intentionally wider
than the final answer set (5) — fusion and reranking need material to
work with. Querying only top-5 from each would leave fusion with too
little to merge meaningfully.

#### 3.2 Reciprocal Rank Fusion — `src/retrieval/fusion.py`

Merges the two result lists using the standard RRF formula:
RRF_score(chunk) = Σ 1 / (k + rank)

where `k=60` (conventional default) and rank is the chunk's position in
each retriever's result list (0-indexed).

**Why RRF and not score averaging:** Dense retrieval returns L2 distance
(~0.3-1.5 range for normalized BGE vectors) and BM25 returns TF-IDF
scores (unbounded, often 0-30). These scales are incompatible — averaging
them directly lets whichever scale is larger dominate arbitrarily. RRF
only uses rank position, making it scale-invariant by design.

**What fusion rewards:** A chunk appearing in both lists at high ranks
gets double contribution to its RRF score. This consensus signal — "both
semantic and keyword search agree this chunk is relevant" — is what makes
hybrid retrieval outperform either method alone.

#### 3.3 Cross-Encoder Reranking — `src/retrieval/reranker.py`

Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to rerank the top-15 fused
candidates down to final top-5.

**Why reranking after fusion:** Both dense and sparse retrieval compute
query and chunk representations **independently** — the query vector is
computed once, chunk vectors are pre-computed, and similarity is a
distance calculation between pre-computed vectors. A cross-encoder reads
query + chunk text **concatenated as one input**, allowing its attention
mechanism to directly model relevance between the specific query and the
specific chunk text. This is more accurate but cannot be precomputed,
which is why it's applied only to the small fused shortlist rather than
the full 611-chunk corpus.

**Score direction:** Cross-encoder scores are raw logits — higher means
more relevant. After sorting descending, top-5 are returned as
`final_results` with `rerank_score` attached.

#### 3.4 RAGPipeline — `src/pipeline.py`

The central class wiring everything together:

```python
pipeline = RAGPipeline()           # loads all models once

# retrieval only (for eval)
result = pipeline.retrieve(query)
# → {"dense_results", "sparse_results", "fused_results", "final_results"}

# full pipeline (for app)
result = pipeline.query(query)
# → above + {"answer"}
```

`retrieve()` returns all intermediate stage outputs deliberately — the
eval harness reads `dense_results`, `sparse_results`, and `final_results`
from the same call to compare three configurations without running the
pipeline three times.

**"Load once, reuse" pattern:** `RAGPipeline.__init__` loads BGE embedder
(~440MB), BM25 index, and cross-encoder (~90MB) at construction time.
In Streamlit, the pipeline is wrapped in `@st.cache_resource` so models
survive across user interactions and page reruns. Without this, every
Streamlit script rerun (triggered on every interaction) would reload all
models from disk — adding 30-60 seconds of dead time per query.

---

### Stage 4 — Generation

#### 4.1 Prompt Design — `src/generation/prompt_templates.py`

The system prompt enforces three hard constraints on the LLM:

1. **Context-only answers** — "Answer only from the provided context
   chunks. Do not use outside knowledge."
2. **Mandatory citations** — "Cite every factual claim using [N] notation.
   Include a References section."
3. **Honest refusal** — "If the context does not contain enough
   information, say exactly: 'The provided context does not contain
   sufficient information to answer this question.'"

Constraint 3 is the most important for RAG correctness. Without it, LLMs
default to confident generation from training knowledge when context is
thin — producing fluent but ungrounded answers. This was verified
empirically: asking "What is the capital of France?" with the system
prompt active correctly produces "not found in context" rather than
"Paris."

The context block formats each chunk as:
[N] Source: paper.pdf, Page 4, Section: Methodology
<chunk text>

The `[N]` numbering maps directly to inline citations in the answer,
making provenance fully traceable.

#### 4.2 LLM Client — `src/generation/llm_client.py`

Wraps OpenRouter's OpenAI-compatible API. Key settings:

- `temperature=0.1` — low temperature for faithful, deterministic
  generation. RAG answers should be consistent and grounded, not creative.
- `max_tokens=1024` — sufficient for research answers without excessive
  cost
- Retry with exponential backoff on 429 rate limit errors — free-tier
  models on OpenRouter have TPM limits that trigger under load

**Model choice rationale:** The LLM's role in RAG is reading comprehension
and structured output — not knowledge recall. A model with a December 2023
knowledge cutoff works perfectly because the relevant knowledge comes from
the retrieved chunks, not the model's training. What matters is
instruction-following quality (does it stay in context?) and citation
format compliance (does it produce `[N]` inline citations?).

---

## Evaluation

Evaluation is structured as a two-phase pipeline in `eval/`, each
independently runnable:

```bash
python -m eval.run_eval                    # both phases
python -m eval.run_eval --retrieval-only   # Phase 2 only (no LLM calls)
python -m eval.run_eval --generation-only  # Phase 3 only
python -m eval.run_eval --use-cached       # reprint report, no new calls
```

### QA Testset — `data/eval/qa_testset.json`

28 hand-authored questions across 4 types:

| Type | Count | Purpose |
|---|---|---|
| Factual | 10 | Direct fact lookup from specific chunks |
| Methodological | 8 | Technical depth — architecture, loss functions |
| Comparative | 5 | Cross-paper synthesis |
| Corpus-absent | 5 | Tests "not found in context" refusal |

**Ground truth chunk IDs** were verified by running each question through
the live pipeline and manually confirming which retrieved chunk actually
contains the answer — not inferred from paper reading. This matters
because chunk IDs depend on the exact `chunk_size` and `chunk_overlap`
settings; assumed IDs from reading the paper would produce false negatives
in the eval.

### Phase 2 — Retrieval Metrics (`eval/retrieval_metrics.py`)

Computes **Recall@k** and **MRR** for three configurations using the same
`pipeline.retrieve()` call — no models reloaded between configs:

| Config | Key accessed |
|---|---|
| Dense only | `result["dense_results"]` |
| Sparse only | `result["sparse_results"]` |
| Hybrid | `result["final_results"]` |

**Recall@k:** Binary — 1.0 if any `relevant_chunk_id` appears in top-k,
0.0 otherwise. Evaluated at k=1, 3, 5 simultaneously.

**MRR:** `1 / rank` of the first relevant chunk. More sensitive than
Recall@k — penalizes systems that find the right chunk at rank 5 vs rank 1.

### Phase 3 — Generation Metrics (`eval/generation_metrics.py`)

Uses RAGAS with two metrics that require **no reference answers**:

**Faithfulness:** Decomposes each generated answer into atomic claims and
checks each against the retrieved context using an LLM judge. Detects
hallucination — answers that go beyond what the context supports.

**Answer Relevancy:** Generates reverse questions from the answer and
checks if they match the original query. Detects topic drift — answers
that are grounded but don't actually address what was asked.

**Sequential scoring:** RAGAS's default parallel async mode hammers
free-tier API rate limits. `run_ragas_sequential()` scores one question
at a time with a configurable delay (`delay_between=15.0s`), trading
runtime (~12 minutes for 25 questions) for reliability. Scores are cached
to `eval/results/ragas_scores_cache.json` — re-runs load from cache
without making any LLM calls.

**RAGAS judge setup:** RAGAS requires an `openai.OpenAI` client instance.
Even when using Groq as the backend, we pass `openai.OpenAI(base_url=
"https://api.groq.com/openai/v1")` rather than the `groq.Groq` client —
RAGAS performs an `isinstance(client, OpenAI)` type check that rejects
the Groq SDK client despite both exposing the same API surface.

BGE embeddings are injected as the RAGAS embedding model via a custom
`BGEEmbeddings(BaseRagasEmbeddings)` subclass — preventing RAGAS from
falling back to OpenAI embeddings (which requires `OPENAI_API_KEY`).

---

## Results

### Retrieval Evaluation (25 answerable questions)

| Config | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| Dense only | 0.4000 | 0.5600 | 0.6000 | 0.4961 |
| Sparse only | 0.2000 | 0.2800 | 0.4000 | 0.2958 |
| **Hybrid (ours)** | **0.4400** | **0.6400** | **0.7600** | **0.5547** |

Hybrid retrieval improves Recall@5 by **+16 points** over dense-only and
**+36 points** over sparse-only.

Dense outperforms sparse on every metric, consistent with the corpus
being semantically rich academic text where meaning/paraphrase matters
more than exact keyword matching.

### Generation Evaluation (25/25 scored, RAGAS LLM-as-judge)

| Metric | Score | Status |
|---|---|---|
| Faithfulness | 0.8241 | Acceptable |
| Answer Relevancy | 0.8882 | Strong |

3 answers scored below 0.5 faithfulness — all correspond to documented
retrieval failures where the relevant chunk did not surface in top-5,
causing the model to either correctly refuse or draw from training
knowledge.

### Why Hybrid Beats Either Method Alone — Concrete Example

For the query *"How does monocular depth estimation compare to stereo?"*:

- Dense search returned `martins2018.pdf_00390` at rank 1 (correct,
  strong semantic match)
- Sparse search returned `1812.11671v1.pdf_00004` at rank 1 (correct,
  strong keyword match)
- Hybrid fusion surfaced `Monocular_Depth_Estimation_A_Thorough_Review_
  00488` at rank 2 — a chunk that **neither** dense nor sparse found
  independently, which contains a direct comparison of monocular vs stereo
  accuracy across distance ranges

This cross-paper synthesis capability — finding relevant content that
neither single-method retriever surfaces alone — is the core value of
hybrid retrieval.

---

## Models & Technology Stack

| Component | Model / Library | Reason for choice |
|---|---|---|
| PDF parsing | `pymupdf4llm` | Reliable markdown output, two-column layout handling |
| Chunking | LangChain `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` | Structure-aware section boundaries |
| Embeddings | `BAAI/bge-base-en-v1.5` | Strong on technical text, free, local |
| Dense index | `ChromaDB` (persistent) | Simple API, metadata filtering, persists between sessions |
| Sparse index | `rank_bm25` (BM25Okapi) | Pure Python, no dependencies, serializable |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast 6-layer model, good precision/latency tradeoff |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) | Scale-invariant, no hyperparameter tuning needed |
| Generation LLM | OpenRouter (`gpt-oss-120b:free`) | Free-tier access to multiple models but gpt-oss was primarily used |
| Generation eval judge | Groq (`llama-3.3-70b-versatile`, `gpt-oss-120b`, `qwen/qwen3-32b`) | Low latency, OpenAI-compatible |
| RAGAS embeddings | `BAAI/bge-base-en-v1.5` (local) | Avoids OpenAI API dependency for eval |
| UI | Streamlit | Rapid prototyping, native chat components |
| Deployment | Streamlit Community Cloud | Free, GitHub-integrated |

---

## Key Design Decisions

**1. Ingestion is a one-time batch job; retrieval is the hot path.**
`build_index.py` runs once and persists both indexes to disk. The query
path reads from these pre-built indexes — it never runs parsing, chunking,
or embedding of documents. This separation means ingestion cost (minutes)
is paid once; retrieval cost (seconds) is paid per query.

**2. "Load once, reuse" for all models.**
`RAGPipeline.__init__` loads BGE embedder, BM25 index, and cross-encoder
at construction time. In Streamlit, `@st.cache_resource` ensures these
survive across reruns. In eval, one pipeline instance is constructed before
the eval loop and reused for all three config comparisons. Never
instantiating models inside a loop or per-query is the single biggest
latency optimization.

**3. Split retrieval and generation in the UI.**
The Streamlit app calls `pipeline.retrieve()` first (2-4s), shows source
chunks immediately, then calls LLM generation (5-15s). Users see retrieved
evidence while the answer is still generating — perceived latency drops
significantly even though total time is unchanged.

**4. `retrieve()` returns all intermediate stage outputs.**
`pipeline.retrieve()` returns `dense_results`, `sparse_results`,
`fused_results`, and `final_results` in one dict. The eval harness reads
whichever key it needs for each config without re-running retrieval. The
Streamlit UI displays the pipeline breakdown (`15 dense + 15 sparse → 28
fused → 5 reranked`) directly from these outputs.

**5. Eval caching at every stage.**
Generated LLM answers: cached to `data/eval/generated_responses.json`.
RAGAS scores: cached to `eval/results/ragas_scores_cache.json`. Retrieval
results: saved to `eval/results/retrieval_results.json`. Re-runs reload
from cache — a full eval (both phases) is re-runnable at zero API cost
using `python -m eval.run_eval --use-cached`.

**6. Ground truth chunk IDs are verified, not assumed.**
`qa_testset.json` relevant chunk IDs were verified by running each
question through the live pipeline and confirming which returned chunk
actually contains the answer. Assumed IDs (based on reading paper content)
would be wrong because chunk IDs depend on exact chunking parameters —
leading to false negatives in Recall@k that look like retrieval failures
but are actually eval setup errors.

---

## Known Limitations

**Retrieval failures (7/25 questions, 28%)** cluster into three documented
categories:

1. **Introductory content retrieval** — Questions about a paper's
   motivation or problem statement hit chunks in the paper's introduction,
   which use generic vocabulary ("we propose," "existing methods suffer
   from") that competes with many other chunks. Dense and sparse search
   both struggle to distinguish "the motivation of *this specific paper*"
   from similar motivation language across the corpus.

2. **Named entity / abbreviation mismatch** — Queries using "FCN" or
   "network architecture" don't match chunks using "Fully Convolutional
   Network based on VGG" because BM25 requires exact token overlap and
   BGE embeddings treat abbreviations and expansions similarly but not
   identically. Query expansion (HyDE or synonym injection) would address
   this.

3. **Single-domain corpus confusion** — All 8 papers discuss monocular
   depth estimation. Paper-specific questions ("what does *this paper*
   propose") compete against similar content across all papers. Metadata
   filtering (`where={"source_file": "paper.pdf"}`) would resolve this
   for paper-specific queries.

**Generation faithfulness (3/25 answers below 0.5):**
All three low-faithfulness answers correspond directly to the retrieval
failures above — when the correct chunk isn't in top-5, the model either
correctly refuses or draws from training knowledge (which RAGAS correctly
penalizes). The faithfulness score is not independent of retrieval quality.

**Hardware constraints:**
The cross-encoder reranker runs on CPU (i5-5300U, no GPU). Reranking 15
candidates takes 1-3 seconds per query. On GPU this would be <200ms.

---

## Running Locally

### Prerequisites

```bash
git clone https://github.com/Ibtesam-Hussain/FYP-research-assistant.git
cd FYP-research-assistant
pip install -r requirements.txt
```

### Environment setup

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_key_here
GROQ_API_KEY=your_key_here       # optional, for eval judge
HF_TOKEN=your_token_here         # optional, avoids HuggingFace rate limits
```

### Running the app

The vector index (`data/chroma_db/`) and BM25 index (`data/bm25_index.pkl`)
are committed to this repo — no indexing step is needed to run the app:

```bash
streamlit run app/streamlit_app.py
```

### Rebuilding indexes from source PDFs

If you add new papers to `data/raw/`:

```bash
python -m src.indexing.build_index --force
```

### Running evaluation

```bash
# retrieval metrics only (no LLM calls, ~2 minutes)
python -m eval.run_eval --retrieval-only

# full eval including RAGAS generation metrics (~15 minutes)
python -m eval.run_eval

# reprint cached results without any new API calls
python -m eval.run_eval --use-cached
```

---

## Deployment

Deployed on Streamlit Community Cloud. The indexes are committed to the
repo (10.5MB ChromaDB + 1.9MB BM25) since Streamlit Cloud has no
persistent storage and cannot run the indexing pipeline on deployment.

API keys are stored in Streamlit Cloud's Secrets manager, not in the repo.

Note: First load takes 30-60 seconds on Streamlit's free-tier CPU while
BGE embedder and cross-encoder download and initialize. Subsequent queries
reuse cached models instantly via `@st.cache_resource`.

---

## About

Built as a portfolio project demonstrating production-oriented AI
engineering practices:

- End-to-end pipeline from raw PDFs to deployed web app
- Quantitative evaluation with honest failure analysis
- Architectural decisions grounded in measurable tradeoffs
- Clean, modular codebase where each component has a single responsibility

**Author:** Ibtesam Hussain — BS Artificial Intelligence, FAST-NUCES Karachi (2026)

**Corpus:** 8 academic papers on monocular and stereo depth estimation
(~611 chunks after processing)

**Live demo:** [fyp-research-assistant.streamlit.app](https://fyp-research-assistant-a4gb5kde2p24dphfc8pgtz.streamlit.app/)