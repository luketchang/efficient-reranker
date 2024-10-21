import torch
from torch.utils.data import Dataset
from data_utils import load_hits_from_qrels_queries_corpus, strip_prefixes

class QueryPassagePairDataset(Dataset):
    def __init__(self, queries_path, corpus_path, rank_results_path, tokenizer, max_seq_len=None):
        self.tokenizer = tokenizer
        rank_results = load_hits_from_qrels_queries_corpus(rank_results_path, queries_path, corpus_path)
        self.max_seq_len = max_seq_len
        self.truncation = max_seq_len is not None

        self.pairs = []
        for rank_result in rank_results:
            hits = rank_result['hits']
            qid = hits[0]['qid']
            for hit in hits:
                self.pairs.append({
                    "qid": qid,
                    "query": rank_result['query'],
                    "pid": hit['docid'],
                    "passage": hit['content'],
                })

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
    
    def collate_fn(self, batch):
        qids = [int(strip_prefixes(item['qid'])) for item in batch]
        pids = [int(strip_prefixes(item['pid'])) for item in batch]
        queries = [item['query'] for item in batch]
        passages = [item['passage'] for item in batch]

        tokenized_pairs = self.tokenizer(queries, passages, padding=True, truncation=self.truncation, return_tensors="pt", max_length=self.max_seq_len)

        return {
            "qids": torch.tensor(qids),
            "pids": torch.tensor(pids),
            "pairs": tokenized_pairs,
        }
    