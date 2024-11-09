from data_utils import load_qid_to_pid_to_score
import argparse

def main(rank_results_path, positives_path, top_k_perc=0.95):
    hits = load_qid_to_pid_to_score(rank_results_path)
    positive_scores = load_qid_to_pid_to_score(positives_path)
    
    gt_pids = {}  # qid --> set of ground truth pids
    gt_min_scores = {}  # qid --> minimum score among ground truth pids

    positives_removed = 0
    for qid, pid_to_score in positive_scores.items():
        gt_pids[qid] = set(pid_to_score.keys())
        min_score = float('inf')
        for pid in gt_pids[qid]:
            score = positive_scores[qid].get(pid, None)
            if score is not None and score < min_score:
                min_score = score
        gt_min_scores[qid] = min_score

        for pid in gt_pids[qid]:
            if pid in hits[qid]:
                positives_removed += 1
                del hits[qid][pid]

    false_negatives_removed = 0
    for qid, pid_to_score in hits.items():
        min_score = gt_min_scores.get(qid, float('inf'))
        for pid, score in list(pid_to_score.items()):
            if pid in gt_pids.get(qid, set()):
                continue  # Do not delete ground truth pids

            print(f"Score: {score}, Min score: {min_score}, Top k perc: {top_k_perc * min_score}")
            if score >= top_k_perc * min_score:
                false_negatives_removed += 1
                del hits[qid][pid]

    output_path = rank_results_path.replace(".tsv", "_negatives.tsv")
    with open(output_path, 'w') as f:
        for qid, pid_to_score in hits.items():
            for pid, score in pid_to_score.items():
                f.write(f"{qid}\t{pid}\t{score}\n")

    print(f"Filtered hits saved to {output_path}")
    print(f"{false_negatives_removed} false negatives removed")
    print(f"{positives_removed} positives removed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove false negatives from hits.")
    parser.add_argument("--rank_results_path", required=True, help="Path to the qrels file")
    parser.add_argument("--positives_path", required=True, help="Path to the ground truth file")
    parser.add_argument("--top_k_perc", type=float, default=0.95, help="Percentage of the minimum ground truth score to keep")
    
    args = parser.parse_args()
    main(args.rank_results_path, args.positives_path, top_k_perc=args.top_k_perc)