import numpy as np
import argparse
from pymilvus import MilvusClient
from data_utils import load_pids_to_passages

def create_index(client, collection_name):
    # Define the index parameters
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="vector", 
        index_type="FLAT", # TODO: will this be too slow?
        metric_type="COSINE",
        params={"nlist": 1024}
    )
    
    # Create the index on the "vector" field
    client.create_index(collection_name=collection_name, index_params=index_params)
    print("Index created.")

# Load vectors from text files
def load_vectors_from_txt(vectors_file):
    return np.loadtxt(vectors_file, delimiter=',')

# Load doc_ids from text files
def load_doc_ids_from_txt(doc_ids_file):
    return np.loadtxt(doc_ids_file, dtype=int)

# Insert data into Milvus
def insert_into_milvus(client, collection_name, doc_ids, vectors, passages, batch_size=100):
    # Loop through doc_ids and vectors in batches
    for i in range(0, len(doc_ids), batch_size):
        batch_doc_ids = doc_ids[i:i + batch_size]
        batch_vectors = vectors[i:i + batch_size]
        
        # Prepare the data batch for Milvus insertion
        data_batch = [
            {
                "id": int(doc_id),
                "vector": vector.tolist(),
                "text": passages.get(str(int(doc_id))),
            }
            for doc_id, vector in zip(batch_doc_ids, batch_vectors)
        ]
        
        # Insert the batch into Milvus
        print(f"Inserting batch {i} vectors into Milvus...")
        client.insert(collection_name=collection_name, data=data_batch)
        print(f"Inserted {len(batch_doc_ids)} vectors into Milvus.")

def main():
    parser = argparse.ArgumentParser(description="Milvus Lite Vector Ingestion CLI")
    parser.add_argument('--corpus_file', type=str, required=True, help="Path to the corpus JSON file containing passages")
    parser.add_argument("--collection_name", type=str, required=True, help="Name of the Milvus collection")
    parser.add_argument('--milvus_db_path', type=str, required=True, help="Path to the Milvus Lite database")
    parser.add_argument('--top_process_index', type=int, required=True, help="Top machine index to process (starts from 0)")
    
    args = parser.parse_args()

    # Load passages from the corpus file
    passages = load_pids_to_passages(args.corpus_file)
    
    # Start Milvus Lite client
    client = MilvusClient(args.milvus_db_path)
    
    # Create Milvus collection (assuming all vectors have the same dimension)
    vectors_file_sample = f"vectors_0.txt"
    vectors_sample = load_vectors_from_txt(vectors_file_sample)
    client.create_collection(collection_name=args.collection_name, dimension=vectors_sample.shape[1])

    # Loop through all machine indices from 0 to top_process_index
    for machine_idx in range(args.top_process_index + 1):
        doc_ids_file = f"doc_ids_{machine_idx}.txt"
        vectors_file = f"vectors_{machine_idx}.txt"
        
        # Load doc_ids and vectors
        doc_ids = load_doc_ids_from_txt(doc_ids_file)
        vectors = load_vectors_from_txt(vectors_file)
        
        # Insert data into Milvus
        insert_into_milvus(client, args.collection_name, doc_ids, vectors, passages)

    # Create index
    print("Creating index...")
    create_index(client, args.collection_name)

    # Make sure we can load collection
    print("Loading the collection...")
    client.load_collection(args.collection_name)
    print(f"Collection '{args.collection_name}' loaded and ready for querying.")


if __name__ == "__main__":
    main()
