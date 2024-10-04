import json
import argparse
from pymilvus import connections, Collection
from embed_utils import get_embedding_model

def main():
    parser = argparse.ArgumentParser(description="Perform similarity search on Milvus and output qrels.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the embedding model")
    parser.add_argument("--queries_file_path", type=str, required=True, help="Path to the queries JSONL file")
    parser.add_argument('--collection_name', type=str, required=True, help='Milvus collection name')
    parser.add_argument("--k", type=int, required=True, help="Number of results to return for each query")
    parser.add_argument("--output_path", type=str, default="qrels.tsv", help="Path to output the qrels.tsv file")
    parser.add_argument('--milvus_host', type=str, default='127.0.0.1', help='Milvus host')
    parser.add_argument('--milvus_port', type=str, default='19530', help='Milvus port')
    parser.add_argument("--batch_size", type=int, default=10, help="Number of queries to process in a batch")

    args = parser.parse_args()

    model_name = args.model_name
    queries_file_path = args.queries_file_path
    collection_name = args.collection_name
    k = args.k
    output_path = args.output_path
    milvus_host = args.milvus_host
    milvus_port = args.milvus_port
    batch_size = args.batch_size

    # Connect to Milvus
    connections.connect(host=milvus_host, port=milvus_port)
    
    # Load the embedding model
    embedding_model, _ = get_embedding_model(model_name)
    
    # Load the Milvus collection
    collection = Collection(collection_name)
    collection.load()

    # Load queries
    query_lines = []
    with open(queries_file_path, 'r') as file:
        for line in file:
            query_lines.append(json.loads(line))

    # Process queries in batches
    query_vectors = []
    query_ids = []
    
    for i, line in enumerate(query_lines):
        query_id = line["_id"]
        query_text = line["text"]
        query_vector = embedding_model.encode(query_text)
        query_vectors.append(query_vector)
        query_ids.append(query_id)

        # If batch size is reached, perform search
        if len(query_vectors) == batch_size:
            print(f"Processing batch {i // batch_size + 1}...")
            results_batch = collection.search(
                data=query_vectors,
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": 32}},
                limit=k,
                output_fields=["id"]
            )

            # Write results for the current batch
            with open(output_path, 'a') as file:
                for i, results in enumerate(results_batch):
                    for hit in results:
                        file.write(f"{query_ids[i]}\t{hit.id}\t{hit.distance}\n")

            # Clear the batch
            query_vectors = []
            query_ids = []

    # Handle any remaining queries that didn't fill the batch
    if query_vectors:
        results_batch = collection.search(
            data=query_vectors,
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": 32}},
            limit=k,
            output_fields=["id"]
        )

        with open(output_path, 'a') as file:
            for i, results in enumerate(results_batch):
                for hit in results:
                    file.write(f"{query_ids[i]}\t{hit.id}\t{hit.distance}\n")

    print(f"Results have been saved to {output_path}")

if __name__ == "__main__":
    main()
