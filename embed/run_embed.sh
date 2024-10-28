#!/bin/bash

export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1  # Force Python to use unbuffered output

# Redirect all output to both a file and stdout
exec > >(tee -a debug_output.log) 2>&1

echo "Starting script at $(date)"
echo "Current working directory: $PWD"
echo "Python path: $PYTHONPATH"

PYTHONPATH=$PWD accelerate launch embed/embed.py \
    --model_name BAAI/bge-en-icl \
    --input_path data/hotpotqa/corpus.jsonl \
    --collection_name hotpot_corpus \
    --batch_size 8

echo "Script finished at $(date)"
