from torch.utils.data import Dataset
from datasets.utils import load_data_from_jsonl, DatasetType
from data_utils import strip_prefixes

class NvEmbedDataset(Dataset):
    def __init__(self, dataset_type: DatasetType, input_path, benchmark, max_seq_len=4096, start_line=0, max_lines=None, prefix_examples=None, qrels_filter_path=None):
        self.dataset_type = dataset_type
        self.input_path = input_path
        self.max_seq_len = max_seq_len
        self.max_lines = max_lines
        self.qrels_filter_path = qrels_filter_path
        self.benchmark = benchmark

        self._load_examples_prefix(prefix_examples)
        self.data = load_data_from_jsonl(dataset_type, input_path, qrels_filter_path, start_line, max_lines)

    def get_query_prefix(self, benchmark):
        instruct = None
        if benchmark == "fiqa":
            instruct = "Given a financial question, retrieve relevant passages that answer the query"
        elif benchmark == "nq":
            instruct = "Given a question, retrieve passages that answer the question"
        elif benchmark == "hotpotqa":
            instruct = "Given a multi-hop question, retrieve documents that can help answer the question"
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        
        return "Instruct: " + instruct +"\nQuery: "

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        id = sample["id"]

        prefix = self.get_query_prefix(self.benchmark) if self.dataset_type == DatasetType.QUERY else ""

        return {
            "id": id,
            "text": text,
            "prefix": prefix
        }
    
    def collate_fn(self, batch):
        ids = [sample['id'] for sample in batch]
        texts = [sample['text'] for sample in batch]
        prefixes = [sample['prefix'] for sample in batch]

        return {
            "ids": ids,
            "texts": texts,
            "prefixes": prefixes
        }
