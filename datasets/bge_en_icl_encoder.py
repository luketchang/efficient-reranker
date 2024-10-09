import torch
from torch.utils.data import Dataset
from enum import Enum
from datasets.utils import load_data_from_jsonl, DatasetType

class BgeEnIclDataset(Dataset):
    def __init__(self, dataset_type: DatasetType, input_path, tokenizer, max_seq_len=4096, start_line=0, max_lines=None, prefix_examples=None, qrels_filter_path=None):
        self.dataset_type = dataset_type
        self.input_path = input_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_lines = max_lines
        self.task = 'Given a query, retrieve relevant passages that answer the query.'
        self.qrels_filter_path = qrels_filter_path

        self._load_examples_prefix(prefix_examples)
        self.data = load_data_from_jsonl(dataset_type, input_path, qrels_filter_path, start_line, max_lines)

    def get_detailed_example(self, query: str, response: str) -> str:
        return f'<instruct>{self.task}\n<query>{query}\n<response>{response}'
    
    def get_detailed_instruct(self, query: str) -> str:
        return f'<instruct>{self.task}\n<query>{query}'

    def _load_examples_prefix(self, examples):
        if examples is None:
            examples = [
                {'instruct': self.task,
                'query': 'what is a virtual interface',
                'response': "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes."},
                {'instruct': self.task,
                'query': 'causes of back pain in female for a week',
                'response': "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management."}
                ]            
        examples = [self.get_detailed_example(e['query'], e['response']) for e in examples]
        self.examples_prefix = '\n\n'.join(examples) + '\n\n' 

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
        # Extract the elements in the batch
        ids = [sample['id'] for sample in batch]
        texts = [sample['text'] for sample in batch]

        max_len = self.max_seq_len
        if self.dataset_type == DatasetType.QUERY:
            texts = self.get_detailed_instruct(texts)
            max_len, texts = self.get_new_queries(texts)
            
        tokenized = self.tokenizer(
            texts,
            max_length=max_len,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),  # Convert ids to tensor
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    # TODO: forgot about get_detailed_instruct
    def get_new_queries(self, queries):
        inputs = self.tokenizer(
            queries,
            max_length=self.max_seq_len - len(self.tokenizer('<s>', add_special_tokens=False)['input_ids']) - len(
                self.tokenizer('\n<response></s>', add_special_tokens=False)['input_ids']),
            return_token_type_ids=False,
            truncation=True,
            return_tensors=None,
            add_special_tokens=False
        )
        prefix_ids = self.tokenizer(self.examples_prefix, add_special_tokens=False)['input_ids']
        suffix_ids = self.tokenizer('\n<response>', add_special_tokens=False)['input_ids']
        new_max_length = (len(prefix_ids) + len(suffix_ids) + self.max_seq_len + 8) // 8 * 8 + 8
        new_queries = self.tokenizer.batch_decode(inputs['input_ids'])
        for i in range(len(new_queries)):
            new_queries[i] = self.examples_prefix + new_queries[i] + '\n<response>'
        return new_max_length, new_queries
