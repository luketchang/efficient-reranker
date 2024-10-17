
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

# get top 1000 embedding rank results (DONE)
# send all embedding rank results to reranker (DONE)
# remove false negatives (within 95% of lowest ground truth rerank score) (DONE)

# need script that loads all query<>positive scores and sends to reranker for score (TODO)
# pass positive qid->pid->score to dataset so it can put positive score onto rank result (TODO)

# Pipeline: 
# - qrels, queries, corpus --> positive rankings (DONE: existing nv_rerank script)
# - corpus, queries --> top 1000 embed (DONE: embed/query scripts)
# - top 1000 embed --> top 200 reranked (DONE: nv_rerank script)
# - top 200 rerank, positive rankings --> top 200 reranked w/out false negatives (DONE: remove_false_negatives script)
# - top 200 reranked w/out false negatives, positive rankings --> teacher triples (TODO)

# NOTE: we don't remove false negatives from rerank stage because we may still want to observe their behavior when sent through reranker or measure scoring

from torch.utils.data import Dataset
from data_utils import load_qid_to_pid_to_score, load_pids_to_passages, load_hits_from_qrels_queries_corpus

class TeacherTriplesDataset(Dataset):
    def __init__(self, queries_path, corpus_path, negative_rank_results_path, positive_rank_results_path):
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
