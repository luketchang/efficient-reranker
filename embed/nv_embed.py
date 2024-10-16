import argparse
import requests
import os
import sys
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, MilvusClient
from datasets.raw_text import RawTextDataset
from datasets.utils import DatasetType
from torch.utils.data import DataLoader
from tqdm import tqdm

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

def fetch_embeddings(api_url, api_key, input_texts):
    payload = {
        "model": "nvidia/nv-embedqa-mistral-7b-v2",
        "encoding_format": "float",
        "truncate": "END",
        "input": input_texts,
        "input_type": "passage",
        "user": "placeholder"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }
    
    response = requests.post(api_url, json=payload, headers=headers)
    response.raise_for_status()  # Raise an error if the request fails
    
    data = response.json()["data"]
    embeddings = [item["embedding"] for item in data]  # List comprehension to extract embeddings
    
    return embeddings

def process_and_insert_data(api_url, api_key, dataloader, client, collection_name, flush_interval=32):
    buffered_data = []
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Processing batches")):
        print(f"Processing batch {batch_idx}")

        pids = batch["ids"]
        texts = batch["texts"]

        # Call the API to fetch embeddings
        vectors = fetch_embeddings(api_url, api_key, texts)

        # Prepare data for insertion
        data = [
            {"id": pids[i], "vector": vectors[i], "text": texts[i]}
            for i in range(len(pids))
        ]
        buffered_data.extend(data)

        # Insert to Milvus in batches
        if (batch_idx + 1) % flush_interval == 0:
            client.insert(collection_name, buffered_data)
            print(f"Inserted {len(buffered_data)} items into the collection")
            buffered_data.clear()

    # Insert remaining data
    if buffered_data:
        client.insert(collection_name, buffered_data)
        print(f"Inserted {len(buffered_data)} remaining items into the collection")

def main():
    parser = argparse.ArgumentParser(description='Embedding insertion script using API and Milvus')
    parser.add_argument('--api_url', type=str, default="https://integrate.api.nvidia.com/v1/embeddings", help='API URL for embedding generation')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--collection_name', type=str, required=True, help='Milvus collection name')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for embedding and insertion')
    parser.add_argument('--milvus_host', type=str, default='127.0.0.1', help='Milvus host')
    parser.add_argument('--milvus_port', type=str, default='19530', help='Milvus port')
    parser.add_argument('--flush_interval', type=int, default=32, help='Number of batches to buffer before flushing to Milvus')
    parser.add_argument('--dim', type=int, required=True, help="Embedding dim")
    args = parser.parse_args()

    api_key = os.environ.get('NGC_API_KEY')
    if not api_key:
        print("Error: NGC_API_KEY environment variable is not set.")
        sys.exit(1)

    # Connect to Milvus
    print(f"Connecting to Milvus at {args.milvus_host}:{args.milvus_port}")
    connections.connect(host=args.milvus_host, port=args.milvus_port)
    client = MilvusClient(f"http://{args.milvus_host}:{args.milvus_port}")

    # Load the dataset
    print(f"Loading dataset from '{args.input_path}'...")
    dataset = RawTextDataset(DatasetType.DOC, args.input_path)  # No tokenizer needed
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    # Create the Milvus collection
    print(f"Creating collection '{args.collection_name}'...")
    create_collection(client, args.collection_name, dim=args.dim)  # Assume embedding dimension is 768
    print("Collection creation complete.")

    # Process the dataset and insert embeddings into Milvus
    print(f"Processing file '{args.input_path}'")
    process_and_insert_data(args.api_url, api_key, dataloader, client, args.collection_name, flush_interval=args.flush_interval)
    print("Data processing complete.")

if __name__ == "__main__":
    main()
