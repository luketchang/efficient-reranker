import torch
from torch.utils.data import Dataset
from data_utils import load_hits_from_rank_results_queries_corpus, load_qid_to_pid_to_score

class QueryPassagePairDataset(Dataset):
    def __init__(self, queries_paths, corpus_paths, rank_results_paths, qrels_paths, tokenizer, qid_bases, max_seq_len=None, hits_per_query=100):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncation = max_seq_len is not None
        self.pairs = []
        self.qrels = []

        # Loop through each dataset
        for queries_path, corpus_path, rank_results_path, qrels_path, qid_base in zip(queries_paths, corpus_paths, rank_results_paths, qrels_paths, qid_bases):
            # Load rank results and ground truth for each dataset
            rank_results = load_hits_from_rank_results_queries_corpus(rank_results_path, queries_path, corpus_path, qrels_filter_path=qrels_path, qid_base=qid_base)
            ground_truth_pid_to_qid_to_score = load_qid_to_pid_to_score(qrels_path, is_qrels=True)
            self.qrels.append(ground_truth_pid_to_qid_to_score)

            # Process each rank result
            for rank_result in rank_results:
                hits = rank_result['hits']
                qid = hits[0]['qid']
                
                for i, hit in enumerate(hits):
                    if i >= hits_per_query:
                        break
                    
                    # Determine if the passage is positive
                    is_positive = qid in ground_truth_pid_to_qid_to_score and hit['docid'] in ground_truth_pid_to_qid_to_score[qid]

                    # Append query-passage pairs with label
                    self.pairs.append({
                        "qid": qid,
                        "query": rank_result['query'],
                        "pid": hit['docid'],
                        "passage": hit['content'],
                        "is_positive": int(is_positive)
                    })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
    
    def collate_fn(self, batch):
        qids = [item['qid'] for item in batch]
        pids = [item['pid'] for item in batch]
        labels = [item['is_positive'] for item in batch]
        queries = [item['query'] for item in batch]
        passages = [item['passage'] for item in batch]

        tokenized_pairs = self.tokenizer(queries, passages, padding=True, truncation=self.truncation, return_tensors="pt", max_length=self.max_seq_len)

        return {
            "qids": qids,
            "pids": pids,
            "labels": torch.tensor(labels),
            "pairs": tokenized_pairs,
        }
    
    def get_qrels(self, i):
        return self.qrels[i]
