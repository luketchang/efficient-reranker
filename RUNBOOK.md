## Creating Train Split

- map all train qrels to test qrels (qrels --> qrels_mapped)
- create subsplit of `n` qrels and queries (qrels_mapped, rank_results_mapped, queries --> qrels_sampled, rank_results_sampled, queries_sampled)
- rerank initial rank_results for those `n` queries (rank_results, queries_sampled, corpus --> --> rerank_results)
- rank positives for those `n` queries (rank_results, queries_sampled, corpus--> positive_rank_results)
- filter out false negatives using positive_rank_results and rerank_results (rerank_results, positive_rank_results --> negative_rank_results)
