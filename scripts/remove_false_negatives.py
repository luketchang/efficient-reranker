from data_utils import load_qid_to_pid_to_score
import argparse

def main(rank_results_path, ground_truth_path, top_k_perc=0.95):
    hits = load_qid_to_pid_to_score(rank_results_path)
    ground_truth_labels = load_qid_to_pid_to_score(ground_truth_path)
    
    gt_pids = {}  # qid --> set of ground truth pids
    gt_min_scores = {}  # qid --> minimum score among ground truth pids

    for qid, pid_to_score in ground_truth_labels.items():
        gt_pids[qid] = set(pid_to_score.keys())
        min_score = float('inf')
        for pid in gt_pids[qid]:
            score = hits[qid].get(pid, None)
            if score is not None and score < min_score:
                min_score = score
        gt_min_scores[qid] = min_score

    for qid, pid_to_score in hits.items():
        min_score = gt_min_scores.get(qid, float('inf'))
        for pid, score in list(pid_to_score.items()):
            if pid in gt_pids.get(qid, set()):
                continue  # Do not delete ground truth pids
            if score >= top_k_perc * min_score:
                del hits[qid][pid]

    output_path = rank_results_path.replace(".tsv", "_filtered.tsv")
    with open(output_path, 'w') as f:
        for qid, pid_to_score in hits.items():
            for pid, score in pid_to_score.items():
                f.write(f"{qid}\t{pid}\t{score}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove false negatives from hits.")
    parser.add_argument("--rank_results_path", required=True, help="Path to the qrels file")
    parser.add_argument("--ground_truth_path", required=True, help="Path to the ground truth file")
    
    args = parser.parse_args()
    main(args.rank_results_path, args.ground_truth_path)