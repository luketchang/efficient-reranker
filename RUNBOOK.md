## Creating Train Split

- map all train qrels to test qrels (qrels --> qrels_mapped)
- create subsplit of `n` qrels and queries (qrels_mapped, queries --> qrels_sampled, queries_sampled)
- rerank initial_rank_results for those `n` queries (qrels_sampled, queries_sampled, corpus --> --> rerank_results)
- rank positives for those `n` queries (qrels_sampled, queries_sampled, corpus--> positive_rank_results)
- filter out false negatives using positive_rank_results and rerank_results (rerank_results, positive_rank_results --> negative_rank_results)
