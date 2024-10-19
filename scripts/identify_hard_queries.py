# load ground truth positives
# load hits from rank_results
# identify hits where positive is at hits index > 5
# print list of query ids for the identified hits

from data_utils import load_qid_to_pid_to_score, load_hits_from_qrels_queries_corpus
import argparse
from collections import defaultdict

def main(rank_results_path, queries_path, corpus_path, positive_qrels, bad_rank_threshold=5):
    rank_results = load_hits_from_qrels_queries_corpus(rank_results_path, queries_path, corpus_path)
    positive_scores = load_qid_to_pid_to_score(positive_qrels)
    
    hard_queries = defaultdict(list)
    for rank_result in rank_results:
        qid = rank_result['hits'][0]['qid']
        hits = rank_result['hits']
        if qid in positive_scores:
            for pid in positive_scores[qid]:
                print("PID", pid)
                index = next((i for i, item in enumerate(hits) if item["docid"] == pid), float('inf'))
                print("Index", index)

                if index >= bad_rank_threshold:
                    hard_queries[index].append(qid)
                    # print(qid)
                    break

    # print each hard query
    print("Hard queries:")
    hard_queries = dict(sorted(hard_queries.items()))
    for index, qids in hard_queries.items():
        print(f"Rank Index: {index}")
        for qid in qids:
            print(f"  Query ID: {qid}")

    print(f"Number of hard queries: {len(hard_queries)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify hard queries from rank results.")
    parser.add_argument('--rank_results_path', type=str, required=True, help='Path to the rank results file.')
    parser.add_argument('--queries_path', type=str, required=True, help='Path to the queries file.')
    parser.add_argument('--corpus_path', type=str, required=True, help='Path to the corpus file.')
    parser.add_argument('--positive_qrels', type=str, required=True, help='Path to the positive rank results file.')
    parser.add_argument('--bad_rank_threshold', type=int, default=5, help='Threshold for identifying hard queries.')

    args = parser.parse_args()
    main(args.rank_results_path, args.queries_path, args.corpus_path, args.positive_qrels, args.bad_rank_threshold)
