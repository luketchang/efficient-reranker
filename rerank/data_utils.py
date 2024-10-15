import json

def load_hits_from_qrels_queries_corpus(qrels_file, queries_file, corpus_file=None):
    print(f"Loading qids from '{queries_file}'")
    queries = load_qids_to_queries(queries_file)

    print(f"Loading corpus from '{corpus_file}'")
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
    for qid in sorted(results.keys(), key=lambda x: int(x.replace("test", "").replace("train", "").replace("dev", ""))):  # Sort by numeric qid
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
            data = json.loads(line)
            pid = data["_id"]
            
            # Extract title and text, combining them if the title exists
            title = data.get("title", "")
            text = data["text"]
            passage = title + "\n" + text if title and title.strip() else text
            
            corpus[pid] = passage
    return corpus