import json

def load_hits_from_rank_results_queries_corpus(rank_results_file, queries_file, corpus_file=None, qrels_filter_path=None, n_hits_per_query=None, qid_base=10):
    print(f"Loading qids from '{queries_file}'")
    queries = load_qids_to_queries(queries_file)
    qid_filter = load_qid_to_pid_to_score(qrels_filter_path) if qrels_filter_path is not None else None

    print(f"Loading corpus from '{corpus_file}'")
    corpus = load_pids_to_passages(corpus_file) if corpus_file is not None else None

    # Step 3: Load qrels and combine all data
    results = {}
    with open(rank_results_file, 'r') as f:
        for line in f:
            # Skip if the first line is the header
            if line.startswith("query-id"):
                continue

            qid, docid, score = line.strip().split('\t')
            score = float(score)

            if qid_filter and qid not in qid_filter:
                continue

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
    for qid in sorted(results.keys(), key=lambda x: int(strip_prefixes(x), qid_base)):  # Sort by numeric qid
        sorted_hits = sorted(
            results[qid]['hits'], 
            key=lambda x: -x['score']  # Sort hits by score in descending order
        )

        if n_hits_per_query is not None:
            sorted_hits = sorted_hits[:n_hits_per_query]

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

def load_pids_to_passages(corpus_file, append_title=True):
    corpus = {}
    with open(corpus_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            pid = data["_id"]
            
            # Extract title and text, combining them if the title exists
            passage = data["text"]
            if append_title:
                title = data.get("title", "")
                passage = title + "\n" + passage if title and title.strip() else passage
            
            corpus[pid] = passage
    return corpus

def load_qid_to_pid_to_score(rank_results_file):
    qid_to_pid_to_score = {}
    with open(rank_results_file, 'r') as f:
        for line in f:
            if line.startswith("query-id"):
                continue

            qid, pid, score = line.strip().split('\t')
            score = float(score)
            
            if qid not in qid_to_pid_to_score:
                qid_to_pid_to_score[qid] = {}
            qid_to_pid_to_score[qid][pid] = score
    return qid_to_pid_to_score

def strip_prefixes(id):
    return id.replace("query", "").replace("doc", "").replace("test", "").replace("train", "").replace("dev", "")