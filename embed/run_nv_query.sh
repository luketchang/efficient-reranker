#!/bin/bash

export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1  # Force Python to use unbuffered output

# Redirect all output to both a file and stdout
exec > >(tee -a debug_output.log) 2>&1

echo "Starting script at $(date)"
echo "Current working directory: $PWD"
echo "Python path: $PYTHONPATH"

PYTHONPATH=$PWD python embed/nv_query.py \
    --api_url http://localhost:8000/v1/embeddings \
    --queries_path data/fiqa/queries.jsonl \
    --k 1000 \
    --output_path data/fiqa/nv_embed_all_1000.tsv \
    --collection_name fiqa_corpus_nv \
    --nprobe 1024 \
    --batch_size 8 \
    --milvus_uri $MILVUS_URI \
    --milvus_token $MILVUS_TOKEN \

echo "Script finished at $(date)"
