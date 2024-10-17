from torch.utils.data import Dataset
from datasets.utils import load_data_from_jsonl, DatasetType

class RawTextSingleDataset(Dataset):
    def __init__(self, dataset_type: DatasetType, input_path, start_line=0, max_lines=None, 
                 qrels_filter_path=None, max_seq_len=512):
        self.dataset_type = dataset_type
        self.input_path = input_path
        self.max_lines = max_lines
        self.qrels_filter_path = qrels_filter_path
        self.max_seq_len = max_seq_len  # Store maxlen

        # Load data from the input JSONL file
        self.data = load_data_from_jsonl(dataset_type, input_path, qrels_filter_path, start_line, max_lines)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]

        # Truncate text if maxlen is set and the text exceeds maxlen
        if self.max_seq_len and len(text) > self.max_seq_len:
            text = text[:self.max_seq_len]

        return {
            "id": sample["id"],
            "text": text
        }

    def collate_fn(self, batch):
        ids = [sample['id'] for sample in batch]
        texts = [sample['text'] for sample in batch]

        # Ensure that all texts are truncated to maxlen if specified
        if self.max_seq_len:
            texts = [text[:self.max_seq_len] if len(text) > self.max_seq_len else text for text in texts]

        return {
            "ids": ids,
            "texts": texts
        }
