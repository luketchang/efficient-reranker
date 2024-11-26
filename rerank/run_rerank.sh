#!/bin/bash

export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1  # Force Python to use unbuffered output

# Redirect all output to both a file and stdout
exec > >(tee -a debug_output.log) 2>&1

echo "Starting script at $(date)"
echo "Current working directory: $PWD"
echo "Python path: $PYTHONPATH"

PYTHONPATH=$PWD python rerank/rerank.py \
    --model_name microsoft/deberta-v3-base \
    --precision bf16 \
    --checkpoint_path new-microsoft/deberta-v3-base-margin_mse-step-14278-inference.pth \
    --rank_results_path data/nq/nv_embed_hf_all_1000.tsv  \
    --qrels_path data/nq/qrels/test.tsv \
    --queries_path data/nq/queries.jsonl \
    --corpus_path data/nq/corpus.jsonl \
    --output_path data/nq/latency_deberta_large.tsv \
    --batch_size 100 \
    --hits_per_query 100 \
    --flush_interval 32 \
    --qid_base 10

echo "Script finished at $(date)"
