import json

def load_hits_from_qrels_queries_corpus(qrels_file, queries_file, corpus_file=None):
    # Step 1: Load Queries File
    queries = load_qids_to_queries(queries_file)

    # Step 2: Load Corpus File (if provided)
    corpus = load_pids_to_passages(corpus_file) if corpus_file is not None else None

    # Step 3: Load qrels and combine all data
    results = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            # Skip if the first line is the header
            if line.startswith("query-id"):
                continue

            qid, docid, score = line.strip().split('\t')
            score = float(score)

            # Initialize query entry if not already present
            if qid not in results:
                results[qid] = {'query': queries[qid], 'hits': []}

            # Create a hit entry
            hit = {
                'qid': qid,
                'docid': docid,
                'score': score,
                'content': corpus[docid] if corpus_file is not None else None
            }

            results[qid]['hits'].append(hit)

    # Step 4: Sort the queries by numeric qid and their hits by score
    rank_results = []
    for qid in sorted(results.keys(), key=lambda x: int(x)):  # Sort by numeric qid
        sorted_hits = sorted(
            results[qid]['hits'], 
            key=lambda x: -x['score']  # Sort hits by score in descending order
        )
        rank_results.append({
            'query': results[qid]['query'],
            'hits': sorted_hits
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
