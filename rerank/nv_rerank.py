import argparse
import requests
import sys
import os
from time import sleep, time
from data_utils import load_hits_from_rank_results_queries_corpus
from tqdm import tqdm


def main(api_url, rank_results_file, queries_file, corpus_file, output_file, start=0, end=None, start_entry=0, window_size=512, k=100, sleep_time=4, qid_base=10):
    api_key = os.environ.get('NGC_API_KEY')
    if not api_key:
        print("Error: NGC_API_KEY environment variable is not set.")
        sys.exit(1)

    headers = {
        "authorization": f"Bearer {api_key}",
        "accept": "application/json",
        "content-type": "application/json",
    }

    rank_results = load_hits_from_rank_results_queries_corpus(rank_results_file, queries_file, corpus_file, qid_base=qid_base)

    end = min(end or len(rank_results), len(rank_results))
    start = max(0, min(start, end - 1))

    # Track latency stats
    total_latency = 0
    total_requests = 0

    with open(output_file, 'a') as f:  # Use 'a' to append results incrementally
        
        # For each set of hits for query: take top k hits, filter out any empty passages, iterate through k samples in window sized batches, send batch to API, update hits with new API logits, resort hits by logit score, write results to file
        for idx in tqdm(range(start, end)):
            query = {"text": rank_results[idx]["query"]}
            hits = rank_results[idx]["hits"][:k]  # Keep only the top 100 hits
            hits = [hit for hit in hits if hit["content"].strip() != ""]
            qid = hits[0]["qid"] if hits else None

            if not hits:
                print(f"No hits found for query {idx + 1}")
                continue

            print(f"Processing query {idx + 1} (QID: {qid})")

            query_results = []  # Collect results for the current query
            for batch_start in range(start_entry, len(hits), window_size):
                batch_end = min(batch_start + window_size, len(hits))
                batch_hits = hits[batch_start:batch_end]
                passages = [{"text": hit["content"]} for hit in batch_hits]

                payload = {
                    "model": "nvidia/nv-rerankqa-mistral-4b-v3",
                    "query": query,
                    "passages": passages,
                    "truncate": "END"
                }

                # Initialize backoff parameters
                delay = 4
                max_delay = 16

                while True:
                    try:
                        request_start = time.perf_counter()
                        response = requests.post(api_url, headers=headers, json=payload)
                        response.raise_for_status()
                        response_body = response.json()
                        request_latency = (time.perf_counter() - request_start) * 1000  # Convert to milliseconds
                        
                        # Update latency stats
                        total_latency += request_latency
                        total_requests += 1

                        print(f"Successfully processed query {idx + 1}, batch {batch_start // window_size + 1}")

                        # Collect the results in query_results list
                        for ranking in response_body["rankings"]:
                            original_index = batch_start + ranking["index"]
                            hits[original_index]["score"] = ranking["logit"]
                            hit = hits[original_index]

                            # Store the results in the query_results list
                            query_results.append((qid, hit["docid"], hit["score"]))

                        sleep(sleep_time)
                        break  # Break the loop on success

                    except requests.exceptions.RequestException as e:
                        print(f"Error occurred while processing query {idx + 1} (QID: {qid}), "
                              f"batch {batch_start // window_size + 1}")
                        print(f"Error details: {str(e)}")

                        if delay > max_delay:
                            print(f"Max retries exceeded for query {idx + 1} (QID: {qid}). Exiting.")
                            sys.exit(1)

                        print(f"Retrying in {delay} seconds...")
                        sleep(delay)
                        delay *= 2  # Exponentially increase the delay

            # Sort results by logit score in descending order
            query_results.sort(key=lambda x: x[2], reverse=True)

            # Write all sorted results for the query to the file
            f.writelines(f"{qid}\t{docid}\t{logit}\n" for qid, docid, logit in query_results)
            f.flush()  # Ensure data is written to disk
            print(f"Average ranking latency after {total_requests} requests: {total_latency/total_requests:.2f}ms")

            # Reset start_entry to 0 after the first query
            start_entry = 0

    print(f"Results written to {output_file}")
    print(f"Final average ranking latency: {total_latency/total_requests:.2f}s over {total_requests} requests")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank hits using NVIDIA's API")
    parser.add_argument("--api_url", type=str, default="https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking", help="URL of the reranking API")
    parser.add_argument("--rank_results_path", type=str, required=True, help="Path to the qrels file")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries file")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus file")
    parser.add_argument("--output_path", type=str, default="reranked_results.tsv", help="Path to output the reranked results")
    parser.add_argument("--start", type=int, default=0, help="Starting index of queries to process")
    parser.add_argument("--end", type=int, default=None, help="Ending index of queries to process (exclusive)")
    parser.add_argument("--start_entry", type=int, default=0, help="Starting entry number within the query (batch level)")
    parser.add_argument("--window_size", type=int, default=100, help="Number of hits to process in a single batch")
    parser.add_argument("--k", type=int, default=100, help="Number of hits to keep for each query")
    parser.add_argument("--sleep_time", type=float, default=4, help="Time to sleep between API requests")
    parser.add_argument("--qid_base", type=int, default=10, help="Base of the qid interpreted as int.")
    args = parser.parse_args()

    main(args.api_url, args.rank_results_path, args.queries_path, args.corpus_path, args.output_path, start=args.start, end=args.end, start_entry=args.start_entry, window_size=args.window_size, k=args.k, sleep_time=args.sleep_time, qid_base=args.qid_base)
