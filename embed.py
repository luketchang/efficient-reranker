import argparse
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sentence_transformers import SentenceTransformer

def get_model(model_name, max_seq_len=None):
    """
    Load the sentence transformer model and adjust the max_seq_length if provided.
    """
    model = SentenceTransformer(model_name, trust_remote_code=True)
    if max_seq_len:
        model.max_seq_length = max_seq_len
    embedding_dim = model.get_sentence_embedding_dimension()  # Automatically get the dimension
    return model, embedding_dim

def create_index(collection):
    # Define the index parameters
    index_params = {
        "index_type": "IVF_FLAT",  # or "HNSW", "IVF_PQ" depending on your needs
        "metric_type": "L2",       # or "IP" for inner product
        "params": {"nlist": 128}
    }
    
    # Create the index
    collection.create_index(field_name="embedding", index_params=index_params)
    print("Index created.")

def create_collection(collection_name, dim):
    """
    Create a Milvus collection schema for storing embeddings.
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),  # Embedding dimension
        FieldSchema(name="passage", dtype=DataType.VARCHAR, max_length=8192),
    ]
    schema = CollectionSchema(fields, f"Schema for {collection_name} with embeddings")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created.")
    return collection

def insert_data(collection, pids, vectors, passages):
    """
    Insert data into Milvus collection using a dictionary-based format for each entity.
    """
    data = [
        {"id": pids[i], "vector": vectors[i], "passage": passages[i]}
        for i in range(len(pids))
    ]
    
    # Insert the data as a list of dictionaries
    collection.insert(data)
    print(f"Inserted {len(pids)} records into Milvus.")
    
def process_file_and_insert(collection, model, file_path, batch_size):
    """
    Process the input file in batches, generate embeddings, and insert them into Milvus.
    """
    pids = []
    passages = []
    vectors = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            pid, passage = line.strip().split('\t', 1)
            pids.append(int(pid))  # Convert pid to integer
            passages.append(passage)

            if len(passages) == batch_size:
                # Generate embeddings for the current batch
                print(f"Processing batch {i // batch_size + 1}...")
                embeddings = model.encode(passages)
                vectors.extend(embeddings)

                # Insert the batch into Milvus
                insert_data(collection, pids, vectors, passages)

                # Clear the lists for the next batch
                pids.clear()
                passages.clear()
                vectors.clear()

        # Insert any remaining data that didn't fill up the last batch
        if passages:
            embeddings = model.encode(passages)
            vectors.extend(embeddings)
            insert_data(collection, pids, vectors, passages)

def main():
    parser = argparse.ArgumentParser(description='Milvus embedding script with Sentence Transformers')
    parser.add_argument('--model_name', type=str, required=True, help='Sentence transformer model name')
    parser.add_argument('--input_file_path', type=str, required=True, help='Path to the input file')
    parser.add_argument('--collection_name', type=str, required=True, help='Milvus collection name')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for embedding and insertion')
    parser.add_argument('--max_seq_len', type=int, default=None, help='Maximum sequence length for the model')
    parser.add_argument('--milvus_host', type=str, default='127.0.0.1', help='Milvus host')
    parser.add_argument('--milvus_port', type=str, default='19530', help='Milvus port')
    args = parser.parse_args()

    # Connect to Milvus
    print(f"Connecting to Milvus at {args.milvus_host}:{args.milvus_port}")
    connections.connect(host=args.milvus_host, port=args.milvus_port)

    # Load the sentence transformer model and get embedding dimensions
    print(f"Loading model '{args.model_name}'...")
    model, embedding_dim = get_model(args.model_name, max_seq_len=args.max_seq_len)
    print(f"Loaded model '{args.model_name}' with embedding dimension: {embedding_dim}")

    # Step 1: Create the Milvus collection
    print(f"Creating collection '{args.collection_name}'...")
    collection = create_collection(args.collection_name, embedding_dim)

    # Step 2: Process file and insert data into Milvus in batches
    print(f"Processing file '{args.input_file_path}' and inserting data into Milvus...")
    process_file_and_insert(collection, model, args.input_file_path, args.batch_size)

    # Step 3: Create an index after data insertion
    print("Creating index...")
    create_index(collection)

    # Step 4: Load the collection after creating the index
    print("Loading the collection...")
    collection.load()
    print(f"Collection '{args.collection_name}' loaded and ready for querying.")


if __name__ == "__main__":
    main()
