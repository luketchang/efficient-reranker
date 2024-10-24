import torch
from torch.utils.data import Dataset
from data_utils import load_qid_to_pid_to_score, load_pids_to_passages, load_hits_from_qrels_queries_corpus, strip_prefixes
import random

class PositiveNegativeDataset(Dataset):
    def __init__(self, queries_path, corpus_path, negative_rank_results_path, positive_rank_results_path, tokenizer, max_seq_len=None, num_neg_per_pos=8, seed=43):
        self.tokenizer = tokenizer
        self.positive_rank_results = load_qid_to_pid_to_score(positive_rank_results_path)
        self.corpus = load_pids_to_passages(corpus_path)
        negative_rank_results = load_hits_from_qrels_queries_corpus(negative_rank_results_path, queries_path, corpus_path)
        self.max_seq_len = max_seq_len
        self.truncation = max_seq_len is not None
        self.num_neg_per_pos = num_neg_per_pos  # Number of negatives to sample
        self.seed = seed  # Global seed

        self.negative_rank_results_with_positives = []
        for rank_result in negative_rank_results:
            hits = rank_result['hits']
            qid = hits[0]['qid']
            if qid in self.positive_rank_results:
                for positive_id in self.positive_rank_results[qid]:
                    positive_score = self.positive_rank_results[qid][positive_id]
                    self.negative_rank_results_with_positives.append({
                        "query_id": qid,
                        "query": rank_result['query'],
                        "positive_id": positive_id,
                        "positive_score": positive_score,
                        "hits": hits  # All hits for negative sampling
                    })

        # Create index mapping: [(query_idx, hit_idx)]
        self.index_mapping = []
        for query_idx, rank_result in enumerate(self.negative_rank_results_with_positives):
            num_hits = len(rank_result['hits'])
            self.index_mapping.extend([query_idx for _ in range(num_hits // self.num_neg_per_pos)])

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        query_idx = self.index_mapping[idx]
        rank_result = self.negative_rank_results_with_positives[query_idx]
        query = rank_result['query']

        # Positive passage
        positive_id = rank_result['positive_id']
        positive_passage = self.corpus[positive_id]

        # Use local RNG for sampling negatives conditioned on seed + idx
        local_rng = random.Random(self.seed + idx)

        # Sample negatives deterministically but locally, using local_rng
        negative_candidates = [hit for hit in rank_result['hits'] if hit['docid'] != positive_id]
        hard_negatives = local_rng.sample(negative_candidates, self.num_neg_per_pos)

        return {
            "query_id": rank_result['query_id'],
            "query": query,
            "positive_id": positive_id,
            "positive": positive_passage,
            "positive_label": rank_result['positive_score'],
            "negative_ids": [neg['docid'] for neg in hard_negatives],
            "negatives": [self.corpus[neg['docid']] for neg in hard_negatives],
            "negative_labels": [neg['score'] for neg in hard_negatives]
        }

    def collate_fn(self, batch):
        queries = [item['query'] for item in batch]
        positive_passages = [item['positive'] for item in batch]
        positive_labels = [item['positive_label'] for item in batch]
        negatives_flattened = [neg for item in batch for neg in item['negatives']]
        negative_labels_flattened = [label for item in batch for label in item['negative_labels']]
        
        # Tokenize positives and negatives
        tokenized_positives = self.tokenizer(queries, positive_passages, padding=True, truncation=self.truncation, return_tensors="pt", max_length=self.max_seq_len)
        repeated_queries = [query for query in queries for _ in range(self.num_neg_per_pos)]
        tokenized_negatives = self.tokenizer(repeated_queries, negatives_flattened, padding=True, truncation=self.truncation, return_tensors="pt", max_length=self.max_seq_len)
        
        return {
            "positives": tokenized_positives,
            "positive_labels": torch.tensor(positive_labels),
            "negatives": tokenized_negatives,
            "negative_labels": torch.tensor(negative_labels_flattened)
        }
