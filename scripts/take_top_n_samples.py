import argparse
from data_utils import load_hits_from_qrels_queries_corpus

def main(rank_results_path, queries_path, corpus_path, output_path):
    rank_results = load_hits_from_qrels_queries_corpus(rank_results_path, queries_path, corpus_path)

    # take top 30 hits for each query
    for rank_result in rank_results:
        rank_result['hits'] = rank_result['hits'][:30]

    with open(output_path, 'w') as f:
        for rank_result in rank_results:
            for hit in rank_result['hits']:
                f.write(f"{hit['qid']}\t{hit['docid']}\t{hit['score']}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Take top N samples from rank results")
    parser.add_argument('--rank_results_path', type=str, required=True, help='Path to the rank results file')
    parser.add_argument('--queries_path', type=str, required=True, help='Path to the queries file')
    parser.add_argument('--corpus_path', type=str, required=True, help='Path to the corpus file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output file')

    args = parser.parse_args()
    main(args.rank_results_path, args.queries_path, args.corpus_path, args.output_path)
            