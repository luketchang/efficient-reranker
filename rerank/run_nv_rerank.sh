#!/bin/bash

export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1  # Force Python to use unbuffered output

# Redirect all output to both a file and stdout
exec > >(tee -a debug_output.log) 2>&1

echo "Starting script at $(date)"
echo "Current working directory: $PWD"
echo "Python path: $PYTHONPATH"

PYTHONPATH=$PWD python rerank/nv_rerank.py \
    --api_url http://localhost:8000/v1/ranking \
    --rank_results_path data/nq/nv_embed_hf_all_1000.tsv \
    --queries_path data/nq/queries.jsonl \
    --corpus_path data/nq/corpus.jsonl \
    --output_path data/nq/latency_nv_rerank.tsv \
    --window_size 100 \
    --sleep_time 0 \
    --qid_base 10

echo "Script finished at $(date)"
