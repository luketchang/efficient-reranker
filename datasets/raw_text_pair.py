# load hits from qrels, queries, corpus ({ qid -> { query, hits: [ { qid, docid, score, content } ] } })
# hits --> array [{ qid, docid, score, content }]
# len(dataset): len(array)
# getitem(idx): array[idx]

from torch.utils.data import Dataset
from data_utils import load_hits_from_qrels_queries_corpus

class RawTextPairDataset(Dataset):
    def __init__(self, qrels_path, queries_path, corpus_path):
        hits = load_hits_from_qrels_queries_corpus(qrels_path, queries_path, corpus_path)
        self.hits = [hit for rank_result in hits for hit in rank_result['hits']]

    def __len__(self):
        return len(self.hits)

    def __getitem__(self, idx):
        item = self.hits[idx]
        return {
            "qid": item['qid'],
            "query": item['query'],
            "pid": item['docid'],
            "passage": item['content'],
        }
    
    def collate_fn(self, batch):
        qids = [sample['qid'] for sample in batch]
        queries = [sample['query'] for sample in batch]
        pids = [sample['pid'] for sample in batch]
        passages = [sample['passage'] for sample in batch]

        return {
            "qids": qids,
            "queries": queries,
            "pids": pids,
            "passages": passages
        }