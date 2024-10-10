import argparse
import pandas as pd
from beir.retrieval.evaluation import EvaluateRetrieval

def load_tsv(filepath, is_qrel=False):
    data = pd.read_csv(filepath, sep='\t', header=0, dtype={'query-id': str, 'corpus-id': str, 'score': float})
    result_dict = {}
    for _, row in data.iterrows():
        query_id, corpus_id, score = row['query-id'], row['corpus-id'], row['score']
        
        if is_qrel:
            score = int(score)

        if query_id not in result_dict:
            result_dict[query_id] = {}
        result_dict[query_id][corpus_id] = score
    return result_dict

def main():
    # Set up argparse
    parser = argparse.ArgumentParser(description="Evaluate retrieval results using BEIR metrics.")
    parser.add_argument("--qrels_path", required=True, type=str, help="Path to qrels TSV file")
    parser.add_argument("--rank_results_path", required=True, type=str, help="Path to rank_results TSV file")
    parser.add_argument("--k_values", nargs='+', type=int, default=[1, 5, 10], help="List of k values for evaluation (default: [1, 5, 10])")
    
    args = parser.parse_args()

    print(f"Loading TSV file from {args.qrels_path}")
    qrels = load_tsv(args.qrels_path, True)
    print(f"Loaded {args.qrels_path}")

    print(f"Loading TSV file from {args.rank_results_path}")
    rank_results = load_tsv(args.rank_results_path)
    print(f"Loaded {args.rank_results_path}")

    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, rank_results, args.k_values)

    print(f"NDCG: {ndcg}")
    print(f"MAP: {_map}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")

if __name__ == "__main__":
    main()
