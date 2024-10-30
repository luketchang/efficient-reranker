#!/bin/bash

export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1  # Force Python to use unbuffered output

# Redirect all output to both a file and stdout
exec > >(tee -a debug_output.log) 2>&1

echo "Starting script at $(date)"
echo "Current working directory: $PWD"
echo "Python path: $PYTHONPATH"

PYTHONPATH=$PWD python rerank/rerank.py \
    --model_name microsoft/deberta-v3-large \
    --checkpoint_path new-microsoft/deberta-v3-large-step-10751-inference.pth \
    --rank_results_path data/hotpotqa/bge_en_icl_all_1000.tsv \
    --qrels_path data/hotpotqa/qrels/test.tsv \
    --queries_path data/hotpotqa/queries.jsonl \
    --corpus_path data/hotpotqa/corpus.jsonl \
    --output_path data/hotpotqa/deberta_margin_mse_reranked_top_100.tsv \
    --hits_per_query 100 \
    --flush_interval 32 \
    --qid_base 16

echo "Script finished at $(date)"
