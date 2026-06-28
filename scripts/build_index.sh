#!/usr/bin/env bash


# Build the index for the evaluation suite
python -m src.indexing.build_index "$@"
