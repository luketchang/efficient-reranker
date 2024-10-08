import json

def load_hits_from_qrels_queries_corpus(qrels_file, queries_file, corpus_file=None):
    # Step 1: Load Queries File
    queries = load_qids_to_queries(queries_file)

    # Step 2: Load Corpus File
    corpus = load_pids_to_passages(corpus_file) if corpus_file is not None else None

    # Step 3: Load qrels and combine all data
    results = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            # skip if first line starts with "query-id"
            if line.startswith("query-id"):
                continue

            qid, docid, score = line.strip().split('\t')
            score = float(score)

            # Add to the results dictionary, initializing if necessary
            if qid not in results:
                results[qid] = {'query': queries[qid], 'hits': []}

            # Optionally add document hits if available
            if corpus_file is not None:
                # Add each hit to the corresponding query
                hit = {
                    'qid': qid,
                    'docid': docid,
                    'score': score
                }

                hit['content'] = corpus[docid]

                results[qid]['hits'].append(hit)

    # Step 4: Convert results into list of dictionaries (to match original format)
    rank_results = []
    for qid in results:
        rank_results.append({
            'query': results[qid]['query'],
            'hits': sorted(results[qid]['hits'], key=lambda x: x['score'], reverse=True)
        })

    return rank_results

def load_qids_to_queries(queries_file):
    queries = {}
    with open(queries_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            qid, query = line["_id"], line["text"]
            queries[qid] = query

    return queries

def load_pids_to_passages(corpus_file):
    corpus = {}
    with open(corpus_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            pid, passage = line["_id"], line["text"]
            corpus[pid] = passage

    return corpus