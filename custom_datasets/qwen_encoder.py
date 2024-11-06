import torch
from torch.utils.data import Dataset
from enum import Enum
from custom_datasets.utils import load_data_from_jsonl, DatasetType

class QwenDataset(Dataset):
    def __init__(self, dataset_type: DatasetType, input_path, tokenizer, max_seq_len=4096, start_line=0, max_lines=None, qrels_filter_path=None):
        self.dataset_type = dataset_type
        self.input_path = input_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_lines = max_lines
        self.task = 'Given a query, retrieve relevant passages that answer the query.'
        self.data = []
        self.qrels_filter_path = qrels_filter_path

        self.data = load_data_from_jsonl(dataset_type, input_path, qrels_filter_path, start_line, max_lines)

    def get_detailed_instruct(self, query: str) -> str:
        return f'Instruct: {self.task}\nQuery: {query}'
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        id = sample["id"]

        return {
            "id": id,
            "text": text
        }
    
    def collate_fn(self, batch):
        ids = [sample['id'] for sample in batch]
        texts = [sample['text'] for sample in batch]

        if self.dataset_type == DatasetType.QUERY:
            texts = [self.get_detailed_instruct(text) for text in texts]
            
        tokenized = self.tokenizer(
            texts,
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),  # Convert ids to tensor
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
