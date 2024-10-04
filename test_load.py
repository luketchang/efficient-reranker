from data_utils import load_hits_from_qrels_queries_corpus

# results = load_hits_from_qrels_queries_corpus("data/nq/nq_qrels.tsv", "data/nq/nq_queries.jsonl")
results = load_hits_from_qrels_queries_corpus("data/msmarco/qrels/train.tsv", "data/msmarco/queries.jsonl")
print(results)
print("len", len(results))