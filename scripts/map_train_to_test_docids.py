import json
import argparse
from collections import defaultdict
from data_utils import load_pids_to_passages, load_qid_to_pid_to_score,strip_prefixes

def load_passage_to_pid(corpus_file, n_chars=200):
    passage_to_pid = {}
    with open(corpus_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            pid = data["_id"]
            passage = data["text"][:n_chars]
            passage_to_pid[passage] = pid
    return passage_to_pid

def main(train_qrels_path, train_corpus_path, test_corpus_path, n_chars):
    test_passage_to_pid_n_chars = load_passage_to_pid(test_corpus_path, n_chars)
    train_pid_to_passage = load_pids_to_passages(train_corpus_path, append_title=False)
    train_qid_to_pid_to_score = load_qid_to_pid_to_score(train_qrels_path)

    no_match_count = 0
    mapped_qid_to_pid_to_score = defaultdict(dict)
    for qid in train_qid_to_pid_to_score.keys():
        pid_to_score = train_qid_to_pid_to_score[qid]
        pid = list(pid_to_score.keys())[0]
        score = pid_to_score[pid]
        passage_n_chars = train_pid_to_passage[pid][:n_chars]

        
        maybe_matching_test_pid = test_passage_to_pid_n_chars.get(passage_n_chars)
        if maybe_matching_test_pid:
            mapped_qid_to_pid_to_score[qid][maybe_matching_test_pid] = score
        else:
            no_match_count += 1

    # Print the first 100 samples from the train_qid_to_pid_to_score dictionary
    count = 0
    for qid, pid_to_score in mapped_qid_to_pid_to_score.items():
        print(f"QID: {qid}, PID to Score: {pid_to_score}")
        count += 1
        if count >= 100:
            break

    output_file = train_qrels_path.replace('.tsv', f'_mapped_{n_chars}.tsv')
    with open(output_file, 'w') as f:
        for qid, pid_to_score in sorted(mapped_qid_to_pid_to_score.items(), key=lambda x: int(strip_prefixes(x[0]))):
            for pid, score in (sorted(pid_to_score.items(), key=lambda x: x[1], reverse=True)):
                f.write(f"{qid}\t{pid}\t{score}\n")

    print(f"No match count: {no_match_count}")
    print(f"Saved mapped qrels to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map train docids to test docids.")
    parser.add_argument('--train_qrels_path', type=str, required=True, help='Path to the train qrels file')
    parser.add_argument('--train_corpus_path', type=str, required=True, help='Path to the train corpus file')
    parser.add_argument('--test_corpus_path', type=str, required=True, help='Path to the test corpus file')
    parser.add_argument('--n_chars', type=int, default=512, help='Number of characters to consider for passage')
    
    args = parser.parse_args()
    main(args.train_qrels_path, args.train_corpus_path, args.test_corpus_path, args.n_chars)