from torch.utils.data import Dataset
from custom_datasets.utils import load_data_from_jsonl, DatasetType

class RawTextSingleDataset(Dataset):
    def __init__(self, dataset_type: DatasetType, input_path, start_line=0, max_lines=None, 
                 qrels_filter_path=None):
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
        text = sample["text"]

        return {
            "id": sample["id"],
            "text": text
        }

    def collate_fn(self, batch):
        ids = [sample['id'] for sample in batch]
        texts = [sample['text'] for sample in batch]

        return {
            "ids": ids,
            "texts": texts
        }
