import requests
from data_utils import load_hits_from_qrels_queries_corpus

invoke_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"

headers = {
    "Authorization": "Bearer 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",
    "Accept": "application/json",
}

rank_results = load_hits_from_qrels_queries_corpus("data/fiqa/bge_en_icl_qrels_1000.tsv", "data/fiqa/queries.jsonl", "data/fiqa/corpus.jsonl")

query = { "text": rank_results[0]["query"] }
passages = [{ "text": hit["content"] } for hit in rank_results[0]["hits"]][:500]

payload = {
  "model": "nv-rerank-qa-mistral-4b:1",
  "query": query,
  "passages": passages
}

# # re-use connections
session = requests.Session()

response = session.post(invoke_url, headers=headers, json=payload)

response.raise_for_status()
response_body = response.json()
print(response_body)