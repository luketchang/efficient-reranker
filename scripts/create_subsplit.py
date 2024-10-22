import random
import argparse
import json
from data_utils import load_qid_to_pid_to_score, load_qids_to_queries, strip_prefixes

def main(qrels_path, queries_path, n):
    qrels = load_qid_to_pid_to_score(qrels_path)
    queries = load_qids_to_queries(queries_path)
    
    qrel_keys = list(qrels.keys())
    random.shuffle(qrel_keys)
    shuffled_n_qrel_keys = qrel_keys[:n]

    sampled_qrels = {qid: qrels[qid] for qid in shuffled_n_qrel_keys}
    sampled_queries = {qid: queries[qid] for qid in shuffled_n_qrel_keys}
    
    qrels_output_path = qrels_path.replace(".tsv", f"_sampled_{n}.tsv")
    queries_output_path = queries_path.replace(".jsonl", f"_sampled_{n}.jsonl")
    
    with open(qrels_output_path, 'w') as f:
        for qid, pid_to_score in sorted(sampled_qrels.items(), key=lambda x: int(strip_prefixes(x[0]))):
            for pid, score in sorted(pid_to_score.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{qid}\t{pid}\t{score}\n")
    
    with open(queries_output_path, 'w') as f:
        for qid, query in sorted(sampled_queries.items(), key=lambda x: int(strip_prefixes(x[0]))):
            json_record = {
                "_id": qid,
                "text": query,
                "metadata": {}
            }
            f.write(json.dumps(json_record) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly split qrels and queries.")
    parser.add_argument("--qrels_path", type=str, help="Path to the qrels file.")
    parser.add_argument("--queries_path", type=str, help="Path to the queries file.")
    parser.add_argument("--n", type=int, help="Number of samples to take.")
    
    args = parser.parse_args()
    main(args.qrels_path, args.queries_path, args.n)