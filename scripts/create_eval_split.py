import random
import json
from data_utils import load_qid_to_pid_to_score, load_qids_to_queries, strip_prefixes
import argparse

def main(qrels_path, queries_path, corpus_path, initial_rank_results_path, n_queries, n_hits_per_query):
    qrels = load_qid_to_pid_to_score(qrels_path)
    queries = load_qids_to_queries(queries_path)
    rank_results = load_qid_to_pid_to_score(initial_rank_results_path)

    # Randomly sample n_queries queries
    sampled_queries = random.sample(list(queries.items()), n_queries)
    sampled_qids = sorted([qid for qid, _ in sampled_queries])

    # Filter qrels and rank_results to only include sampled queries
    sampled_queries = {qid: queries[qid] for qid in sampled_qids}
    sampled_qrels = {qid: qrels[qid] for qid in sampled_qids if qid in qrels}
    sampled_rank_results = {qid: rank_results[qid] for qid in sampled_qids if qid in rank_results}

    def to_int(qid):
        return int(strip_prefixes(qid))

    sampled_queries = dict(sorted(sampled_queries.items(), key=lambda x:to_int(x[0])))
    sampled_qrels = dict(sorted(sampled_qrels.items(), key=lambda x: to_int(x[0])))
    sampled_rank_results = dict(sorted(sampled_rank_results.items(), key=lambda x: to_int(x[0])))

    # Write sampled queries, qrels, and rank results to output files
    qrels_output_path = qrels_path.replace(".tsv", f"_sampled_{n_queries}.tsv")
    queries_output_path = queries_path.replace(".jsonl", f"_sampled_{n_queries}.jsonl")
    rank_results_output_path = initial_rank_results_path.replace(".tsv", f"_sampled_{n_queries}.tsv")

    # Write everything to output files
    with open(qrels_output_path, 'w') as f:
        for qid, pid_to_score in sampled_qrels.items():
            for pid, score in pid_to_score.items():
                f.write(f"{qid}\t{pid}\t{score}\n")

    with open(queries_output_path, 'w') as f:
        for qid, query in sampled_queries.items():
            json_record = {
                "_id": qid,
                "text": query,
                "metadata": {}
            }
            f.write(json.dumps(json_record) + "\n")

    with open(rank_results_output_path, 'w') as f:
        for qid, pid_to_score in sampled_rank_results.items():
            num_docs = 0
            for pid, score in pid_to_score.items():
                if num_docs >= n_hits_per_query:
                    break

                f.write(f"{qid}\t{pid}\t{score}\n")
                num_docs += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create evaluation split from qrels, queries, and initial rank results.")
    parser.add_argument("--qrels_path", type=str, required=True, help="Path to the qrels file.")
    parser.add_argument("--queries_path", type=str, required=True, help="Path to the queries file.")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus file.")
    parser.add_argument("--initial_rank_results_path", type=str, required=True, help="Path to the initial rank results file.")
    parser.add_argument("--n_queries", type=int, required=True, help="Number of queries to sample.")
    parser.add_argument("--n_hits_per_query", type=int, required=True, help="Number of hits per query.")

    args = parser.parse_args()

    main(
        qrels_path=args.qrels_path,
        queries_path=args.queries_path,
        corpus_path=args.corpus_path,
        initial_rank_results_path=args.initial_rank_results_path,
        n_queries=args.n_queries,
        n_hits_per_query=args.n_hits_per_query
    )