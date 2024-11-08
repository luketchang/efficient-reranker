import random
import argparse
import json
from data_utils import load_qid_to_pid_to_score, load_qids_to_queries, strip_prefixes

def main(qrels_path, rank_results_path, queries_path, split_amounts, samples_per_query=100, qid_base=10):
    n_total_samples = sum(split_amounts)
    qrels = load_qid_to_pid_to_score(qrels_path)
    rank_results = load_qid_to_pid_to_score(rank_results_path)
    queries = load_qids_to_queries(queries_path)
    
    qrel_keys = list(qrels.keys())
    random.shuffle(qrel_keys)
    shuffled_n_qrel_keys = qrel_keys[:n_total_samples]

    qrels_key_splits = []
    start_idx = 0
    for amount in split_amounts:
        end_idx = start_idx + amount
        qrels_key_splits.append(shuffled_n_qrel_keys[start_idx:end_idx])
        start_idx = end_idx

    sampled_qrels = [{qid: qrels[qid] for qid in key_split} for key_split in qrels_key_splits]
    sampled_rank_results = [{qid: rank_results[qid] for qid in key_split} for key_split in qrels_key_splits]
    sampled_queries = [{qid: queries[qid] for qid in key_split} for key_split in qrels_key_splits]
    
    qrels_output_paths = [qrels_path.replace(".tsv", f"_sampled_{n}.tsv") for n in split_amounts]
    queries_output_paths = [queries_path.replace(".jsonl", f"_sampled_{n}.jsonl") for n in split_amounts]
    rank_results_paths = [rank_results_path.replace(".tsv", f"_sampled_{n}.tsv") for n in split_amounts]
        
    for i in range(len(split_amounts)):
        with open(qrels_output_paths[i], 'w') as f:
            for qid, pid_to_score in sorted(sampled_qrels[i].items(), key=lambda x: int(strip_prefixes(x[0]), qid_base)):
                for pid, score in sorted(pid_to_score.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{qid}\t{pid}\t{score}\n")

        with open(rank_results_paths[i], 'w') as f:
            for qid, pid_to_score in sorted(sampled_rank_results[i].items(), key=lambda x: int(strip_prefixes(x[0]), qid_base)):
                for pid, score in sorted(pid_to_score.items(), key=lambda x: x[1], reverse=True)[:samples_per_query]:
                    f.write(f"{qid}\t{pid}\t{score}\n")
    
        with open(queries_output_paths[i], 'w') as f:
            for qid, query in sorted(sampled_queries[i].items(), key=lambda x: int(strip_prefixes(x[0]), qid_base)):
                json_record = {
                    "_id": qid,
                    "text": query,
                    "metadata": {}
                }
                f.write(json.dumps(json_record) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly split qrels and queries.")
    parser.add_argument("--qrels_path", type=str, help="Path to the qrels file.")
    parser.add_argument("--rank_results_path", type=str, help="Path to the rank results file.")
    parser.add_argument("--queries_path", type=str, help="Path to the queries file.")
    parser.add_argument("--split_amounts", nargs="+", type=int, help="Number of samples to take for each split (e.g. [4000, 1500])")
    parser.add_argument("--samples_per_query", type=int, default=100, help="Number of samples to take for each query.")
    parser.add_argument("--qid_base", type=int, default=10, help="Base of the qid interpreted as int.")
    
    args = parser.parse_args()
    main(args.qrels_path, args.rank_results_path, args.queries_path, args.split_amounts, args.samples_per_query, args.qid_base,)