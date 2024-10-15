import argparse
import requests
from pymilvus import MilvusClient, connections
from datasets.raw_text import RawTextDataset
from datasets.utils import DatasetType
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

def fetch_query_embeddings(api_url, api_key, queries):
    payload = {
        "model": "nvidia/nv-embedqa-mistral-7b-v2",
        "encoding_format": "float",
        "truncate": "NONE",
        "input": queries,
        "input_type": "query",
        "user": "placeholder"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    response = requests.post(api_url, json=payload, headers=headers)
    response.raise_for_status()  # Raise error if the request fails

    data = response.json()["data"]
    embeddings = [item["embedding"] for item in data]
    
    return embeddings

def write_results_to_file(output_path, query_ids, results_batch, qrels_qid_prefix="", qrels_pid_prefix=""):
    """Write search results to the output file in TSV format."""
    with open(output_path, 'a') as file:
        for i, results in enumerate(results_batch):
            for hit in results:
                qid = qrels_qid_prefix + str(query_ids[i])
                pid = qrels_pid_prefix + str(hit['entity']['id'])
                file.write(f"{qid}\t{pid}\t{hit['distance']}\n")

def main():
    parser = argparse.ArgumentParser(description="Perform similarity search on Milvus using API-generated query embeddings.")
    parser.add_argument("--api_url", type=str, default="https://integrate.api.nvidia.com/v1/embeddings", help="API URL for embedding generation")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries JSONL file")
    parser.add_argument("--qrels_filter_path", type=str, help="Path to the qrels TSV file to filter out queries")
    parser.add_argument("--collection_name", type=str, required=True, help="Milvus collection name")
    parser.add_argument("--k", type=int, required=True, help="Number of results to return for each query")
    parser.add_argument("--output_path", type=str, default="qrels.tsv", help="Path to the qrels output file")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of queries to process in a batch")
    parser.add_argument('--milvus_host', type=str, default='127.0.0.1', help='Milvus host')
    parser.add_argument('--milvus_port', type=str, default='19530', help='Milvus port')
    parser.add_argument("--qrels_qid_prefix", type=str, default="", help="Prefix for query IDs in qrels file")
    parser.add_argument("--qrels_pid_prefix", type=str, default="", help="Prefix for passage IDs in qrels file")

    args = parser.parse_args()

    api_key = os.environ.get('NGC_API_KEY')
    if not api_key:
        print("Error: NGC_API_KEY environment variable is not set.")
        sys.exit(1)

    # Connect to Milvus
    print(f"Connecting to Milvus at {args.milvus_host}:{args.milvus_port}")
    connections.connect(host=args.milvus_host, port=args.milvus_port)
    client = MilvusClient(f"http://{args.milvus_host}:{args.milvus_port}")
    print(f"Loading collection '{args.collection_name}'...")
    client.load_collection(args.collection_name)

    # Load queries dataset
    print(f"Loading queries from '{args.queries_path}'...")
    dataset = RawTextDataset(DatasetType.QUERY, args.queries_path, qrels_filter_path=args.qrels_filter_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    print(f"Loaded {len(dataset)} queries.")

    # Process queries and perform similarity search
    print(f"Processing queries and retrieving top {args.k} results...")
    for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        print(f"Processing batch {i}")
        query_ids = batch["ids"]
        queries = batch["text"]

        # Fetch query embeddings using the API
        query_vectors = fetch_query_embeddings(args.api_url, api_key, queries)

        # Perform similarity search in Milvus
        results_batch = client.search(
            collection_name=args.collection_name,
            data=query_vectors,
            anns_field="vector",
            search_params={"metric_type": "IP", "params": {"nprobe": 512}},
            limit=args.k,
            output_fields=["id"]
        )

        # Write the search results to the output file
        write_results_to_file(args.output_path, query_ids, results_batch, args.qrels_qid_prefix, args.qrels_pid_prefix)

    print(f"Results have been saved to {args.output_path}")

if __name__ == "__main__":
    main()
