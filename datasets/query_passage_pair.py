import torch
from torch.utils.data import Dataset
from data_utils import load_hits_from_rank_results_queries_corpus, load_qid_to_pid_to_score
import hashlib

class QueryPassagePairDataset(Dataset):
    def __init__(self, queries_path, corpus_path, rank_results_path, qrels_path, tokenizer, max_seq_len=None, hits_per_query=100):
        self.tokenizer = tokenizer
        rank_results = load_hits_from_rank_results_queries_corpus(rank_results_path, queries_path, corpus_path)
        ground_truth_pid_to_qid_to_score = load_qid_to_pid_to_score(qrels_path)
        self.max_seq_len = max_seq_len
        self.truncation = max_seq_len is not None

        self.pairs = []
        for rank_result in rank_results:
            hits = rank_result['hits']
            qid = hits[0]['qid']
            for i, hit in enumerate(hits):
                if i >= hits_per_query:
                    break
                
                is_positive =  qid in ground_truth_pid_to_qid_to_score and hit['docid'] in ground_truth_pid_to_qid_to_score[qid]

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
        # NOTE: for evaluation, we need qids and pids even in collate_fn so we hash the strings which should contain unique tags for each dataset (e.g. nq, hotpotqa)
        def hash_id(id_str):
            return int(hashlib.md5(id_str.encode()).hexdigest(), 16) % (2**63 - 1)

        hashed_qids = [hash_id(item['qid']) for item in batch]
        hashed_pids = [hash_id(item['pid']) for item in batch]
        labels = [item['is_positive'] for item in batch]
        queries = [item['query'] for item in batch]
        passages = [item['passage'] for item in batch]

        tokenized_pairs = self.tokenizer(queries, passages, padding=True, truncation=self.truncation, return_tensors="pt", max_length=self.max_seq_len)

        return {
            "qids": torch.tensor(hashed_qids, dtype=torch.int64),
            "pids": torch.tensor(hashed_pids, dtype=torch.int64),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "pairs": tokenized_pairs,
        }
    