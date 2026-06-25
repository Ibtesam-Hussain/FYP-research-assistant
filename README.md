# FYP Research Assistant

A document retrieval and generation pipeline for academic papers and report QA.

## Structure

- `data/raw/`: original PDFs
- `data/processed/`: parsed and chunked output
- `data/eval/`: evaluation dataset and results
- `src/`: ingestion, indexing, retrieval, generation, and pipeline code
- `app/`: Streamlit UI
- `eval/`: evaluation configuration and metrics
- `notebooks/`: exploratory analysis
- `scripts/`: helper scripts
- `tests/`: unit tests

## Usage

1. Populate `data/raw/` with your PDFs.
2. Run `scripts/build_index.sh`.
3. Run `scripts/run_eval.sh`.
4. Launch `app/streamlit_app.py` with Streamlit.
