#!/bin/bash

export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1  # Force Python to use unbuffered output

# Redirect all output to both a file and stdout
exec > >(tee -a debug_output.log) 2>&1

echo "Starting script at $(date)"
echo "Current working directory: $PWD"
echo "Python path: $PYTHONPATH"

PYTHONPATH=$PWD python embed/query.py \
    --model_name BAAI/bge-en-icl \
    --queries_path data/hotpotqa/queries.jsonl \
    --k 1000 \
    --output_path data/hotpotqa/bge_en_icl_all_1000.tsv \
    --collection_name hotpot_corpus \
    --batch_size 8 \
    --milvus_uri $MILVUS_URI \
    --milvus_token $MILVUS_TOKEN \

echo "Script finished at $(date)"
