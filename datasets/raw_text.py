from torch.utils.data import Dataset
from datasets.utils import load_data_from_jsonl, DatasetType

class RawTextDataset(Dataset):
    def __init__(self, dataset_type: DatasetType, input_path, start_line=0, max_lines=None, qrels_filter_path=None):
        self.dataset_type = dataset_type
        self.input_path = input_path
        self.max_lines = max_lines
        self.qrels_filter_path = qrels_filter_path

        # Load data from the input JSONL file
        self.data = load_data_from_jsonl(dataset_type, input_path, qrels_filter_path, start_line, max_lines)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "id": sample["id"],
            "text": sample["text"]
        }

    def collate_fn(self, batch):
        ids = [sample['id'] for sample in batch]
        texts = [sample['text'] for sample in batch]

        if self.dataset_type == DatasetType.QUERY:
            texts = [f'Instruct: Retrieve relevant passages.\nQuery: {text}' for text in texts]

        return {
            "ids": ids,
            "texts": texts
        }