import argparse
import torch
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, MilvusClient
from transformers import AutoModel, AutoTokenizer
from datasets.bge_en_icl_encoder import BgeEnIclDataset
from datasets.utils import DatasetType
from torch.utils.data import DataLoader
from accelerate import Accelerator, DeepSpeedPlugin
from embed_utils import last_token_pool
from tqdm import tqdm

def create_index(client, collection_name):
    # Define the index parameters
    index_params = client.prepare_index_params()

    # Create the index on the "vector" field
    index_params.add_index(
        field_name="vector", 
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={ "nlist": 128 }
    )
    
    client.create_index(collection_name, index_params=index_params)

def create_collection(client, collection_name, dim):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=32_768),
    ]
    schema = CollectionSchema(fields, f"Schema for {collection_name} with embeddings")
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )

def encode_data(accelerator, model, dataloader, tokenizer, client, collection_name, flush_interval=32):
    buffered_data = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing batches")):
        accelerator.print(f"Processing batch {batch_idx}")

        inputs = {k: v.to(accelerator.device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        pids = batch["ids"]
        texts = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

        with torch.no_grad():
            outputs = model(**inputs)
            vectors = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"]).to(torch.float32)

        # NOTE: we keep data on GPU here (we batch move to CPU later)
        data = [
            {"id": pids[i], "vector": vectors[i], "text": texts[i]}
            for i in range(len(pids))
        ]
        buffered_data.extend(data)

        # If buffer has reached flush_interval batches, move to CPU and insert all to client
        if (batch_idx + 1) % flush_interval == 0:
            for item in buffered_data:
                item["vector"] = item["vector"].cpu().numpy()

            client.insert(collection_name, buffered_data)
            accelerator.print(f"Inserted {len(buffered_data)} items into the collection")
            buffered_data.clear()
            torch.cuda.empty_cache()

        # Free up memory
        del pids, vectors, outputs, inputs, texts, data

    # After all batches, if any data left in buffer, insert it
    if buffered_data:
        for item in buffered_data:
            item["vector"] = item["vector"].cpu().numpy()

        client.insert(collection_name, buffered_data)
        accelerator.print(f"Inserted {len(buffered_data)} remaining items into the collection")
        buffered_data.clear()

    accelerator.print(f"Process {accelerator.process_index} finished encoding data")

def main():
    parser = argparse.ArgumentParser(description='Milvus embedding script with Sentence Transformers')
    parser.add_argument('--model_name', type=str, required=True, help='Sentence transformer model name')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument("--start_line", type=int, default=0, help="Starting line number to read from the input file")
    parser.add_argument('--max_input_lines', type=int, default=None, help='Maximum number of lines to read from the input file')
    parser.add_argument('--collection_name', type=str, required=True, help='Milvus collection name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for embedding and insertion')
    parser.add_argument('--max_seq_len', type=int, default=None, help='Maximum sequence length for the model')
    parser.add_argument('--milvus_host', type=str, default='127.0.0.1', help='Milvus host')
    parser.add_argument('--milvus_port', type=str, default='19530', help='Milvus port')
    parser.add_argument("--use_ds", type=bool, default=False, help="Use DeepSpeed for training")
    parser.add_argument('--flush_interval', type=int, default=32, help='Number of batches to buffer before flushing to Milvus')
    args = parser.parse_args()

    # Connect to Milvus
    print(f"Connecting to Milvus at {args.milvus_host}:{args.milvus_port}")
    connections.connect(host=args.milvus_host, port=args.milvus_port)
    client = MilvusClient(f"http://{args.milvus_host}:{args.milvus_port}")

    # Setup accelerator
    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=1,           # ZeRO stage 1 for inference?
        offload_optimizer_device="none",
        offload_param_device="none",
    ) if args.use_ds else None
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, device_placement=True)
    accelerator.print(f"State: {accelerator.state}")

    # Load the dataset to embed
    accelerator.print(f"Loading dataset from '{args.input_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = BgeEnIclDataset(DatasetType.DOC, args.input_path, tokenizer, max_seq_len=args.max_seq_len, start_line=args.start_line, max_lines=args.max_input_lines)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    # Load the model and get embedding dimensions
    accelerator.print(f"Loading model '{args.model_name}'...")
    model = AutoModel.from_pretrained(args.model_name, torch_dtype=torch.float16)
    model.eval()
    embedding_dim = model.config.hidden_size
    accelerator.print(f"Loaded model '{args.model_name}' with embedding dimension: {embedding_dim}")

    # Prepare model and dataloader
    model, dataloader = accelerator.prepare(model, dataloader)

    # Create the Milvus collection (main process only)
    if accelerator.is_main_process:
        accelerator.print(f"Creating collection '{args.collection_name}'...")
        create_collection(client, args.collection_name, embedding_dim)
        accelerator.print("Collection creation complete.")

    # Wait for all processes before proceeding
    accelerator.wait_for_everyone()

    # Process file and insert data into Milvus in batches
    accelerator.print(f"Worker {accelerator.process_index} processing file '{args.input_path}'")
    encode_data(accelerator, model, dataloader, tokenizer, client, args.collection_name, flush_interval=args.flush_interval)
    accelerator.print(f"Data processing complete for process {accelerator.process_index}")

    # Ensure everyone has written data before creating index
    accelerator.wait_for_everyone()

    # Create index (main process only)
    if accelerator.is_main_process:
        create_index(client, args.collection_name)
        accelerator.print("Index creation complete.")

if __name__ == "__main__":
    main()
