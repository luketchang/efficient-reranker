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
    --checkpoint_path new-microsoft/deberta-v3-large-step-25599-inference.pth \
    --rank_results_path data/nq/bge_en_icl_qrels_1000_ip.tsv \
    --qrels_path data/nq/qrels/test.tsv \
    --queries_path data/nq/queries.jsonl \
    --corpus_path data/nq/corpus.jsonl \
    --output_path deberta_info_nce_reranked_top_200.tsv \
    --qid_prefix test \
    --pid_prefix doc

echo "Script finished at $(date)"
