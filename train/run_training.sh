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
    --queries_paths data/nq-train/queries_sampled_6250.jsonl data/hotpotqa/queries_sampled_6250.jsonl \
    --corpus_paths data/nq/corpus.jsonl data/hotpotqa/corpus.jsonl \
    --train_positive_rank_results_paths data/nq-train/nv_rerank_positives_sampled_6250.tsv data/hotpotqa/nv_rerank_positives_sampled_6250.tsv  \
    --train_negative_rank_results_paths data/nq-train/nv_rerank_negatives_top100_sampled_6250_filtered.tsv data/hotpotqa/nv_rerank_negatives_top100_sampled_6250_filtered.tsv \
    --train_qid_bases 10 16 \
    --eval_qrels_paths data/nq/qrels/test_sampled_700.tsv data/hotpotqa/qrels/test_sampled_1500.tsv \
    --eval_rank_results_paths data/nq/bge_en_icl_qrels_1000_ip_sampled_700.tsv data/hotpotqa/bge_en_icl_all_1000_sampled_1500.tsv \
    --eval_queries_paths data/nq/queries_sampled_700.jsonl data/hotpotqa/queries_sampled_1500.jsonl \
    --eval_qid_bases 10 16 \
    --eval_every_n_batches 128 \
    --lr 0.00002 \
    --grad_clip_max_norm 0.6 \
    --grad_accumulation_steps 4 \
    --seed 43 \
    --dropout_prob 0.1

echo "Script finished at $(date)"
