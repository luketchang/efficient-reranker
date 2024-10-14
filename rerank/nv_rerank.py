import argparse
import requests
import sys
import os
from time import sleep
from data_utils import load_hits_from_qrels_queries_corpus
import ftfy
from unidecode import unidecode
import re
from tqdm import tqdm

def clean_text(text):
    # 1. Fix common encoding issues using ftfy
    text = ftfy.fix_text(text)

    # 2. Convert non-ASCII characters to ASCII equivalents
    text = unidecode(text)

    # 3. Replace problematic financial symbols with text equivalents
    text = text.replace('%', ' percent').replace('$', 'USD')

    # 4. Clean up spaces and redundant punctuation
    text = ' '.join(text.split())  # Remove excess spaces
    text = re.sub(r'([.,!?])\1+', r'\1', text)  # Collapse repeated punctuation

    # 5. Replace smart quotes with plain quotes
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')

    return text

def main(qrels_file, queries_file, corpus_file, output_file, start=0, end=None, start_entry=0, window_size=512, sleep_time=4):
    invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/nv-rerankqa-mistral-4b-v3/reranking"

    api_key = os.environ.get('NVIDIA_API_KEY')
    if not api_key:
        print("Error: NVIDIA_API_KEY environment variable is not set.")
        sys.exit(1)

    headers = {
        "authorization": f"Bearer {api_key}",
        "accept": "application/json",
        "content-type": "application/json",
    }

    rank_results = load_hits_from_qrels_queries_corpus(qrels_file, queries_file, corpus_file)

    end = min(end or len(rank_results), len(rank_results))
    start = max(0, min(start, end - 1))

    with open(output_file, 'a') as f:  # Use 'a' to append results incrementally
        for idx in tqdm(range(start, end)):
            query = {"text": rank_results[idx]["query"]}
            hits = rank_results[idx]["hits"]
            qid = hits[0]["qid"]

            print(f"Processing query {idx + 1} (QID: {qid})")

            query_results = []  # Collect results for the current query

            for batch_start in range(start_entry, len(hits), window_size):
                batch_end = min(batch_start + window_size, len(hits))
                batch_hits = hits[batch_start:batch_end]
                passages = [
                    {"text": clean_text(hit["content"])}
                    for hit in batch_hits
                    if clean_text(hit["content"]).strip()  # Exclude empty/whitespace-only strings
                ]

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
                        response = requests.post(invoke_url, headers=headers, json=payload)
                        response.raise_for_status()
                        response_body = response.json()

                        print(f"Successfully processed query {idx + 1}, batch {batch_start // window_size + 1}")

                        # Collect the results in query_results list
                        for ranking in response_body["rankings"]:
                            original_index = batch_start + ranking["index"]
                            hit = hits[original_index]
                            hit["score"] = ranking["logit"]

                            # Store the results in the query_results list
                            query_results.append((qid, hit["docid"], ranking["logit"]))

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

            # Reset start_entry to 0 after the first query
            start_entry = 0

    print(f"Results written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank hits using NVIDIA's API")
    parser.add_argument("--qrels_path", type=str, required=True, help="Path to the qrels file")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries file")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus file")
    parser.add_argument("--output_path", type=str, default="reranked_results.tsv", help="Path to output the reranked results")
    parser.add_argument("--start", type=int, default=0, help="Starting index of queries to process")
    parser.add_argument("--end", type=int, default=None, help="Ending index of queries to process (exclusive)")
    parser.add_argument("--start_entry", type=int, default=0, help="Starting entry number within the query (batch level)")
    parser.add_argument("--window_size", type=int, default=512, help="Number of hits to process in a single batch")
    parser.add_argument("--sleep_time", type=int, default=4, help="Time to sleep between API requests")
    args = parser.parse_args()

    main(args.qrels_path, args.queries_path, args.corpus_path, args.output_path,
         start=args.start, end=args.end, start_entry=args.start_entry, window_size=args.window_size, sleep_time=args.sleep_time)
