from enum import Enum
import json

class DatasetType(Enum):
    QUERY = 0,
    DOC = 1

def load_qrels(qrels_path):
        qids = set()
        with open(qrels_path, 'r') as file:
            for line in file:
                qid = line.strip().split()[0]
                qid = qid.replace("query", "").replace("test", "")
                qids.add(qid)
        return qids

def load_data_from_jsonl(dataset_type, input_path, qrels_filter_path=None, max_lines=None):
        data_arr = []
        qids_filter = set()
        if dataset_type == DatasetType.QUERY and qrels_filter_path:
            qids_filter = load_qrels(qrels_filter_path)
            print(f"Loaded {len(qids_filter)} qids from qrels filter.")

        # Load the data
        with open(input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                data = json.loads(line)
                id = data["_id"].replace("doc", "").replace("test", "")  # Remove prefixes
                
                # Filter queries if QIDs filter is applied
                if dataset_type == DatasetType.QUERY and qids_filter and id not in qids_filter:
                    print("Skipping query", id)
                    continue

                title = data.get("title", "")
                text = data["text"]
                passage = title + " " + text if title else text
                data_arr.append({"id": int(id), "text": passage})

        return data_arr