import json
import argparse
from pymilvus import connections, Collection
from embeddings.sentence_transformers import SentenceTransformersEmbeddingModel

def load_qrels(qrels_filter_path):
    """Load qids from the qrels tsv file into a set."""
    qids = set()
    with open(qrels_filter_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                qid = parts[0]
                qids.add(qid)
    return qids

def load_queries_filtered_by_qrels(queries_path, qrels_filter_path=None):
    """
    Load queries from a JSONL file and filter them by QIDs if qrels_filter_path is provided.

    :param queries_path: Path to the queries JSONL file.
    :param qrels_filter_path: Path to the Qrels file to filter queries, optional.
    :return: A list of queries that are in the Qrels (if applicable).
    """
    # Load Qrels filter if defined
    qids_filter = set()
    if qrels_filter_path:
        qids_filter = load_qrels(qrels_filter_path)
        print(f"Loaded {len(qids_filter)} queries from qrels filter.")

    # Load queries and filter them based on Qrels
    query_lines = []
    with open(queries_path, 'r') as file:
        for line in file:
            query = json.loads(line)
            query_id = query["_id"]
            # Only add the query if it's in the qids_filter (if qrels_filter_path is defined)
            if not qids_filter or query_id in qids_filter:
                query_lines.append(query)

    print(f"Loaded {len(query_lines)} queries for processing.")
    return query_lines

def write_results_to_file(output_path, query_ids, results_batch):
    """Write search results to the output file in tsv format."""
    with open(output_path, 'a') as file:
        for i, results in enumerate(results_batch):
            for hit in results:
                file.write(f"{query_ids[i]}\t{hit.id}\t{hit.distance}\n")

def main():
    parser = argparse.ArgumentParser(description="Perform similarity search on Milvus and output qrels.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the embedding model")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries JSONL file")
    parser.add_argument("--qrels_filter_path", type=str, help="Path to the qrels TSV file that will filter out queries")
    parser.add_argument('--collection_name', type=str, required=True, help='Milvus collection name')
    parser.add_argument("--k", type=int, required=True, help="Number of results to return for each query")
    parser.add_argument("--output_path", type=str, default="qrels.tsv", help="Path to output the qrels.tsv file")
    parser.add_argument('--milvus_host', type=str, default='127.0.0.1', help='Milvus host')
    parser.add_argument('--milvus_port', type=str, default='19530', help='Milvus port')
    parser.add_argument("--batch_size", type=int, default=10, help="Number of queries to process in a batch")

    args = parser.parse_args()

    model_name = args.model_name
    queries_path = args.queries_path
    collection_name = args.collection_name
    k = args.k
    output_path = args.output_path
    milvus_host = args.milvus_host
    milvus_port = args.milvus_port
    batch_size = args.batch_size
    qrels_filter_path = args.qrels_filter_path

    # Load queries and Qrels filtering (if applicable)
    query_lines = load_queries_filtered_by_qrels(queries_path, qrels_filter_path)

    # Connect to Milvus
    connections.connect(host=milvus_host, port=milvus_port)
    
    # Load the embedding model
    embedding_model = SentenceTransformersEmbeddingModel(model_name)
    
    # Load the Milvus collection
    collection = Collection(collection_name)
    collection.load()

    # Process queries in batches
    query_vectors = []
    query_ids = []
    
    for i, line in enumerate(query_lines):
        query_id = line["_id"]
        query_text = line["text"]
        query_vector = embedding_model.encode_queries(query_text)
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
            write_results_to_file(output_path, query_ids, results_batch)

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

        write_results_to_file(output_path, query_ids, results_batch)

    print(f"Results have been saved to {output_path}")

if __name__ == "__main__":
    main()
