import argparse
from collections import defaultdict
from data_utils import load_qid_to_pid_to_score  # Assuming the function is in data_utils

def main(rank_results_path, positive_rank_results_path, bad_rank_threshold=5):
    rank_results = load_qid_to_pid_to_score(rank_results_path)
    positive_scores = load_qid_to_pid_to_score(positive_rank_results_path)

    hard_queries = defaultdict(list)

    for qid in positive_scores:
        hits = rank_results.get(qid, {})
        pid_to_score = {pid: score for pid, score in hits.items()}

        for pid, positive_score in positive_scores[qid].items():
            # Add positive score if missing or update with max score if it exists in the hits
            if pid in pid_to_score:
                pid_to_score[pid] = max(pid_to_score[pid], positive_score)
            else:
                pid_to_score[pid] = positive_score

            # Sort the combined hits by score in descending order
            combined_hits = sorted(pid_to_score.items(), key=lambda x: x[1], reverse=True)
            rank = next((i for i, (p, _) in enumerate(combined_hits) if p == pid), None)
            
            if rank is not None:
                rank += 1  # Ranks start from 1
                if rank >= bad_rank_threshold:
                    hard_queries[rank].append((qid, pid))

                    if rank <= 50:
                        new_value = positive_score * 2 if positive_score > 0 else positive_score / 2
                        positive_scores[qid][pid] = new_value
                        rank_results[qid][pid] = new_value

    # Write back the updated positive scores to the positive rank results file
    new_positive_rank_results_path = positive_rank_results_path.replace('.tsv', '_bumped.tsv')
    with open(new_positive_rank_results_path, 'w') as f:
        for qid, pid_scores in positive_scores.items():
            for pid, score in pid_scores.items():
                f.write(f"{qid}\t{pid}\t{score}\n")

    # Write back the updated rank results to the rank results file
    new_rank_results_path = rank_results_path.replace('.tsv', '_bumped.tsv')
    with open(new_rank_results_path, 'w') as f:
        for qid, pid_scores in rank_results.items():
            for pid, score in pid_scores.items():
                f.write(f"{qid}\t{pid}\t{score}\n")

    # Print hard queries with their PIDs
    print("Hard queries:")
    hard_queries = dict(sorted(hard_queries.items()))
    for rank, qid_pid_pairs in hard_queries.items():
        print(f"Rank Index: {rank}")
        for qid, pid in qid_pid_pairs:
            print(f"  Query ID: {qid}, Passage ID: {pid}")
    print(f"Number of hard queries: {len(hard_queries)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify hard queries from rank results.")
    parser.add_argument('--rank_results_path', type=str, required=True, help='Path to the rank results file.')
    parser.add_argument('--positive_rank_results_path', type=str, required=True, help='Path to the positive scores file.')
    parser.add_argument('--bad_rank_threshold', type=int, default=10, help='Threshold for identifying hard queries.')
    args = parser.parse_args()
    main(args.rank_results_path, args.positive_rank_results_path, args.bad_rank_threshold)
