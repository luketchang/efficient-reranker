import torch
from torch.utils.data import Dataset
from data_utils import load_qid_to_pid_to_score, load_pids_to_passages, load_hits_from_qrels_queries_corpus

class TeacherTriplesDataset(Dataset):
    def __init__(self, queries_path, corpus_path, negative_rank_results_path, positive_rank_results_path, tokenizer):
        self.tokenizer = tokenizer
        self.positive_rank_results = load_qid_to_pid_to_score(positive_rank_results_path)
        self.corpus = load_pids_to_passages(corpus_path)
        negative_rank_results = load_hits_from_qrels_queries_corpus(negative_rank_results_path, queries_path, corpus_path)

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
                        "hits": hits
                    })

        # Create index mapping: [(query_idx, hit_idx)]
        self.index_mapping = []
        for query_idx, rank_result in enumerate(self.negative_rank_results_with_positives):
            num_hits = len(rank_result['hits'])
            self.index_mapping.extend([(query_idx, hit_idx) for hit_idx in range(num_hits)])

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        query_idx, hit_idx = self.index_mapping[idx]
        rank_result = self.negative_rank_results_with_positives[query_idx]
        query = rank_result['query']
        hit = rank_result['hits'][hit_idx]
        return {
            "query": query,
            "positive_id": rank_result['positive_id'],
            "positive": self.corpus[rank_result['positive_id']],
            "positive_score": rank_result['positive_score'],
            "negative_id": hit['docid'],
            "negative": hit['content'],
            "negative_score": hit['score']
        }
    
    def collate_fn(self, batch):
        queries = [item['query'] for item in batch]
        positive_passages = [item['positive'] for item in batch]
        positive_scores = [item['positive_score'] for item in batch]
        negative_passages = [item['negative'] for item in batch]
        negative_scores = [item['negative_score'] for item in batch]

        tokenized_positives = self.tokenizer(queries, positive_passages, padding=True, return_tensors="pt")
        tokenized_negatives = self.tokenizer(queries, negative_passages, padding=True, return_tensors="pt")

        return {
            "positives": tokenized_positives,
            "positive_labels": torch.tensor(positive_scores),
            "negatives": tokenized_negatives,
            "negative_labels": torch.tensor(negative_scores)
        }
