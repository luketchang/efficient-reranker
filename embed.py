import argparse
import os
import numpy as np
import torch
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from transformers import AutoModel, AutoTokenizer
from datasets.instruct_encoder import InstructEncoderDataset, DatasetType
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
from embed_utils import last_token_pool
from tqdm import tqdm
import numpy as np

def encode_data(accelerator, model, dataloader, buffer_num_batches=25):
    doc_ids_file = f"doc_ids_{accelerator.process_index}.txt"
    vectors_file = f"vectors_{accelerator.process_index}.txt"

    # Initialize buffers
    doc_ids_buffer = []
    vectors_buffer = []

    # Wrap the dataloader with tqdm for progress tracking
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing batches")):
        accelerator.print(f"Processing batch {batch_idx}")

        # Filter the inputs to include only 'input_ids' and 'attention_mask'
        inputs = {k: v for k, v in batch.items() if k == "input_ids" or k == "attention_mask"}
        pids = batch["ids"]

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            vectors = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])

        # Accumulate the current batch's data in GPU memory
        doc_ids_buffer.append(pids.detach().cpu())
        vectors_buffer.append(vectors.detach().cpu())

        # If the buffer is large enough, write to file
        if len(doc_ids_buffer) >= buffer_num_batches:
            accelerator.print("Flushing buffer to file")
            flush_buffer_to_file(doc_ids_file, vectors_file, doc_ids_buffer, vectors_buffer)
            doc_ids_buffer = []
            vectors_buffer = []

        # Clear GPU memory
        del pids, vectors, outputs
        torch.cuda.empty_cache()

    # After all batches, write any remaining data in the buffer to file
    if doc_ids_buffer:
        flush_buffer_to_file(doc_ids_file, vectors_file, doc_ids_buffer, vectors_buffer)

    accelerator.print(f"Data saved incrementally to {doc_ids_file} and {vectors_file}")

def flush_buffer_to_file(doc_ids_file, vectors_file, doc_ids_buffer, vectors_buffer):
    doc_ids = [pid.numpy() for pid in doc_ids_buffer]
    vectors = [vector.numpy() for vector in vectors_buffer]

    # Concatenate the buffered data
    doc_ids = np.concatenate(doc_ids, axis=0)
    vectors = np.concatenate(vectors, axis=0)

    # Write the data to the respective files in append mode
    with open(doc_ids_file, 'a') as f_doc, open(vectors_file, 'a') as f_vec:
        np.savetxt(f_doc, doc_ids, fmt='%d')  # Save doc_ids as integers
        np.savetxt(f_vec, vectors, fmt='%.8f')  # Save vectors with 8 decimal precision

    del doc_ids
    del vectors
        
def main():
    parser = argparse.ArgumentParser(description='Milvus embedding script with Sentence Transformers')
    parser.add_argument('--model_name', type=str, required=True, help='Sentence transformer model name')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--max_input_lines', type=int, default=None, help='Maximum number of lines to read from the input file')
    parser.add_argument('--collection_name', type=str, required=True, help='Milvus collection name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for embedding and insertion')
    parser.add_argument('--max_seq_len', type=int, default=None, help='Maximum sequence length for the model')
    parser.add_argument('--milvus_host', type=str, default='127.0.0.1', help='Milvus host')
    parser.add_argument('--milvus_port', type=str, default='19530', help='Milvus port')
    parser.add_argument("--use_ds", type=bool, default=False, help="Use DeepSpeed for training")
    args = parser.parse_args()

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=2,           # Use ZeRO stage 2 (stage 3 offloads even more, but is slower)
        offload_optimizer_device="none",  # Whether to offload optimizer state to CPU (reduce GPU VRAM)
        offload_param_device="none",       # Whether to offload parameters to CPU (reduce GPU VRAM)
        # hf_ds_config=ds_config_path
    ) if args.use_ds else None

    # Setup accelerator
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, device_placement=True)
    accelerator.print(f"State: {accelerator.state}")

    # Load the dataset to embed
    accelerator.print(f"Loading dataset from '{args.input_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = InstructEncoderDataset(DatasetType.DOC, args.input_path, tokenizer, max_seq_len=args.max_seq_len, max_lines=args.max_input_lines)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    # Load the sentence transformer model and get embedding dimensions
    accelerator.print(f"Loading model '{args.model_name}'...")
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=torch.float16)
    model.eval()
    embedding_dim = model.config.hidden_size
    accelerator.print(f"Loaded model '{args.model_name}' with embedding dimension: {embedding_dim}")

    model, dataloader = accelerator.prepare(model, dataloader)

    # Process file and insert data into Milvus in batches
    accelerator.print(f"Processing file '{args.input_path}' and inserting data into txt files for process {accelerator.process_index}")
    encode_data(accelerator, model, dataloader)
    accelerator.print(f"Data processing complete for process {accelerator.process_index}")


if __name__ == "__main__":
    main()