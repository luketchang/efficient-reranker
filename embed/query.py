import argparse
from pymilvus import MilvusClient, connections
from transformers import AutoModel, AutoTokenizer
from datasets.utils import DatasetType
from datasets.bge_en_icl_encoder import BgeEnIclDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from embed_utils import last_token_pool

def write_results_to_file(output_path, query_ids, results_batch, qrels_qid_prefix="", qrels_pid_prefix=""):
    """Write search results to the output file in tsv format."""
    with open(output_path, 'a') as file:
        for i, results in enumerate(results_batch):
            for hit in results:
                qid = qrels_qid_prefix + str(query_ids[i])
                pid = qrels_pid_prefix + str(hit['entity']['id'])
                file.write(f"{qid}\t{pid}\t{hit['distance']}\n")

def main():
    parser = argparse.ArgumentParser(description="Perform similarity search on Milvus and output qrels.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the embedding model")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries JSONL file")
    parser.add_argument("--qrels_filter_path", type=str, help="Path to the qrels TSV file that will filter out queries")
    parser.add_argument("--collection_name", type=str, required=True, help="Milvus collection name")
    parser.add_argument("--k", type=int, required=True, help="Number of results to return for each query")
    parser.add_argument("--nprobe", type=int, default=512, help="Number of probes for the search")
    parser.add_argument("--output_path", type=str, default="qrels.tsv", help="Path to output the qrels.tsv file")
    parser.add_argument("--batch_size", type=int, default=10, help="Number of queries to process in a batch")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="Maximum sequence length for the model")
    parser.add_argument('--milvus_uri', type=str, default='127.0.0.1:19530', help='Milvus uri')
    parser.add_argument('--milvus_token', type=str, default=None, help='Milvus token')
    parser.add_argument("--qrels_qid_prefix", type=str, default="", help="Prefix for the qrels file")
    parser.add_argument("--qrels_pid_prefix", type=str, default="", help="Suffix for the qrels file")

    args = parser.parse_args()

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the tokenizer and dataset for queries
    print(f"Loading tokenizer and dataset from '{args.queries_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    dataset = BgeEnIclDataset(
        dataset_type=DatasetType.QUERY, 
        input_path=args.queries_path, 
        tokenizer=tokenizer, 
        max_seq_len=args.max_seq_len, 
        qrels_filter_path=args.qrels_filter_path
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    print("length of dataset:", len(dataset))

    # connect to Milvus
    print(f"Connecting to Milvus at {args.milvus_uri}")
    connections.connect(alias="default", uri=args.milvus_uri, token=args.milvus_token)
    client = MilvusClient(uri=args.milvus_uri, token=args.milvus_token)
    
    print("Loading collection...")
    client.load_collection(args.collection_name)

    # Load the embedding model
    print(f"Loading embedding model '{args.model_name}'...")
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    # Process queries in batches using DataLoader
    print(f"Processing queries and retrieving top {args.k} results...")
    for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        print(f"Processing batch {i}")
        query_ids = batch["ids"].tolist() if torch.is_tensor(batch["ids"]) else batch["ids"]
        inputs = {k: v.to(device) for k, v in batch.items() if k == "input_ids" or k == "attention_mask"}

        with torch.no_grad():
            # Generate query embeddings
            outputs = model(**inputs)
            query_vectors = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"]).to(torch.float32)

        # Perform search in Milvus for the batch of queries
        results_batch = client.search(
            collection_name=args.collection_name,
            data=query_vectors.cpu().numpy(),  # Move to CPU for Milvus
            anns_field="vector",
            search_params={"metric_type": "IP", "params": {"nprobe": args.nprobe}},
            limit=args.k,
            output_fields=["id"]
        )

        # Write the results to the output file
        write_results_to_file(args.output_path, query_ids, results_batch, args.qrels_qid_prefix, args.qrels_pid_prefix)

    print(f"Results have been saved to {args.output_path}")

if __name__ == "__main__":
    main()
