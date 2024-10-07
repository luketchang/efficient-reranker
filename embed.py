import argparse
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from transformers import AutoModel, AutoTokenizer
from datasets.instruct_encoder import InstructEncoderDataset, DatasetType
from torch.utils.data import DataLoader
from accelerate import Accelerator
from embed_utils import last_token_pool
from tqdm import tqdm

def create_index(collection):
    # Define the index parameters
    index_params = {
        "index_type": "IVF_FLAT",  # or HNSW, IVF_PQ depending on your needs
        "metric_type": "COSINE",
        "params": {"nlist": 3000}
    }
    
    # Create the index on the "vector" field
    collection.create_index(field_name="vector", index_params=index_params)
    print("Index created.")

def create_collection(collection_name, dim):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),  # Embedding dimension
        FieldSchema(name="passage", dtype=DataType.VARCHAR, max_length=8192),
    ]
    schema = CollectionSchema(fields, f"Schema for {collection_name} with embeddings")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created.")
    return collection

def encode_data(accelerator, model, dataloader, tokenizer, collection):
    # Wrap the dataloader with tqdm for progress tracking
    for batch in tqdm(dataloader, desc=f"Processing batches"):
        # Filter the inputs to include only 'input_ids' and 'attention_mask'
        inputs = {k: v for k, v in batch.items() if k == "input_ids" or k == "attention_mask"}
        pids = batch["ids"]

        # Decode input_ids into human-readable text
        passages = tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)

        # Generate embeddings
        outputs = model(**inputs)
        vectors = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])

        # Insert the data into the collection (e.g., database)
        data = [
            {"id": pids[i], "vector": vectors[i], "passage": passages[i]}
            for i in range(len(pids))
        ]
        
        # Insert the data as a list of dictionaries
        collection.insert(data)
        accelerator.print(f"Inserted batch with {len(pids)} items into the collection")


def main():
    parser = argparse.ArgumentParser(description='Milvus embedding script with Sentence Transformers')
    parser.add_argument('--model_name', type=str, required=True, help='Sentence transformer model name')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input JSONL file')
    parser.add_argument('--max_input_lines', type=int, default=None, help='Maximum number of lines to read from the input file')
    parser.add_argument('--collection_name', type=str, required=True, help='Milvus collection name')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for embedding and insertion')
    parser.add_argument('--max_seq_len', type=int, default=None, help='Maximum sequence length for the model')
    parser.add_argument('--milvus_host', type=str, default='127.0.0.1', help='Milvus host')
    parser.add_argument('--milvus_port', type=str, default='19530', help='Milvus port')
    args = parser.parse_args()

    # Connect to Milvus
    print(f"Connecting to Milvus at {args.milvus_host}:{args.milvus_port}")
    connections.connect(host=args.milvus_host, port=args.milvus_port)

    # Setup accelerator
    accelerator = Accelerator(device_placement=True)
    accelerator.print(f"State: {accelerator.state}")

    # Load the dataset to embed
    accelerator.print(f"Loading dataset from '{args.input_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = InstructEncoderDataset(DatasetType.DOC, args.input_path, max_seq_len=args.max_seq_len, max_lines=args.max_input_lines)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load the sentence transformer model and get embedding dimensions
    accelerator.print(f"Loading model '{args.model_name}'...")
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()
    embedding_dim = model.config.hidden_size
    accelerator.print(f"Loaded model '{args.model_name}' with embedding dimension: {embedding_dim}")

    model, dataloader = accelerator.prepare(model, dataloader)

    # Create the Milvus collection (main process only)
    if accelerator.is_main_process:
        accelerator.print(f"Creating collection '{args.collection_name}'...")
        collection = create_collection(args.collection_name, embedding_dim)

    # No processing until main process has finished creating collection
    accelerator.wait_for_everyone() 

    # Process file and insert data into Milvus in batches
    accelerator.print(f"Processing file '{args.input_path}' and inserting data into Milvus...")
    encode_data(collection, model, args.input_path, args.batch_size, max_lines=args.input_max_lines)

    # Create index on the collection (main process only)
    if accelerator.is_main_process:
        accelerator.print("Creating index...")
        create_index(collection)

        print("Loading the collection...")
        collection.load()
        print(f"Collection '{args.collection_name}' loaded and ready for querying.")


if __name__ == "__main__":
    main()