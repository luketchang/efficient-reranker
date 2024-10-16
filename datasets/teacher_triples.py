
# load ground truth positives
# load top 30-50 reranked (negative) hits
# treat dataset as one long list of hits
#  - expand each query into its set of top k hits
#  - len(dataset) = sum(len(hits) for hits in dataset)
#  - getitem(idx) returns (idx // num queries) + (idx % num queries)

# load qid --> queries
# load pid --> passages
# load rank results as qid --> pid --> score
# load ground truth as qid --> pid --> score

from torch.utils.data import Dataset
from data_utils import load_qid_to_pid_to_score, load_pids_to_passages, load_hits_from_qrels_queries_corpus

class TeacherTriplesDataset(Dataset):
    def __init__(self, queries_path, corpus_path, negative_rank_results_path, ground_truth_path):
        self.ground_truth = load_qid_to_pid_to_score(ground_truth_path)
        self.corpus = load_pids_to_passages(corpus_path)
        negative_rank_results = load_hits_from_qrels_queries_corpus(negative_rank_results_path, queries_path, corpus_path)

        negative_rank_results_with_positives = []
        for qid, hits in negative_rank_results.items():
            if qid in self.ground_truth:
                for positive_id in self.ground_truth[qid]:
                    negative_rank_results_with_positives.append({
                        "query": hits['query'],
                        "query_id": qid,
                        "positive_id": positive_id,
                        "hits": hits['hits']
                    })
        self.negative_rank_results_with_positives = negative_rank_results_with_positives
        
    def __len__(self):
        return sum(len(positive_with_negatives['hits']) for positive_with_negatives in self.negative_rank_results_with_positives)
    
    def __getitem__(self, idx):
        outer_idx = idx // len(self.negative_rank_results_with_positives)
        hit_idx = idx % len(self.negative_rank_results_with_positives)
        rank_result = self.negative_rank_results_with_positives[outer_idx]
        query = rank_result['query']
        hit = rank_result['hits'][hit_idx]
        return {
            "query": query,
            "positive": self.corpus[rank_result['positive_id']],
            "negative": hit['content']
        }
    