#!/bin/bash

export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1  # Force Python to use unbuffered output

# Redirect all output to both a file and stdout
exec > >(tee -a debug_output.log) 2>&1

echo "Starting script at $(date)"
echo "Current working directory: $PWD"
echo "Python path: $PYTHONPATH"

accelerate launch train/train_script.py \
    --model_name microsoft/deberta-v3-large \
    --num_epochs 1 \
    --batch_size_per_gpu 4 \
    --queries_path data/nq-train/matching_queries.jsonl \
    --corpus_path data/nq/corpus.jsonl \
    --train_positive_rank_results_path data/nq-train/nv_rerank_positives_sampled_10000.tsv \
    --train_negative_rank_results_path data/nq-train/nv_rerank_negatives_top200_sampled_10000_filtered.tsv \
    --eval_qrels_path data/nq/qrels/test_sampled_200.tsv \
    --eval_rank_results_path data/nq/bge_en_icl_qrels_1000_ip_sampled_200.tsv \
    --eval_queries_path data/nq/queries_sampled_200.jsonl \
    --eval_every_n_batches 512 \
    --lr 0.00002 \
    --grad_clip_max_norm 0.6 \
    --grad_accumulation_steps 4 \
    --seed 43 \
    --dropout_prob 0.2

echo "Script finished at $(date)"
