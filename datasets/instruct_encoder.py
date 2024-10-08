import json
import torch
from torch.utils.data import Dataset
from enum import Enum

class DatasetType(Enum):
    QUERY = 0,
    DOC = 1

class InstructEncoderDataset(Dataset):
    def __init__(self, dataset_type: DatasetType, input_path, tokenizer, max_seq_len=4096, max_lines=None, prefix_examples=None, qrels_filter_path=None):
        self.dataset_type = dataset_type
        self.input_path = input_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_lines = max_lines
        self.task = 'Given a query, retrieve relevant passages that answer the query.'
        self.data = []
        self.qrels_filter_path = qrels_filter_path

        self._load_examples_prefix(prefix_examples)
        self._load_data(qrels_filter_path)
        print("MAX SEQ LEN", self.max_seq_len)

    def _load_qrels(self, qrels_filter_path):
        qids = set()
        with open(qrels_filter_path, 'r') as file:
            for line in file:
                qid = line.strip().split()[0]
                qid = qid.replace("query", "").replace("test", "")
                qids.add(qid)
        return qids

    def _load_data(self, qrels_filter_path=None):
        # Load QIDs filter from qrels if provided
        qids_filter = set()
        if self.dataset_type == DatasetType.QUERY and self.qrels_filter_path:
            qids_filter = self._load_qrels(qrels_filter_path)
            print(f"Loaded {len(qids_filter)} qids from qrels filter.")

        # Load the data
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if self.max_lines and i >= self.max_lines:
                    break
                data = json.loads(line)
                id = data["_id"].replace("doc", "").replace("test", "")  # Remove prefixes
                
                # Filter queries if QIDs filter is applied
                if self.dataset_type == DatasetType.QUERY and qids_filter and id not in qids_filter:
                    print("Skipping query", id)
                    continue

                title = data.get("title", "")
                text = data["text"]
                passage = title + " " + text if title else text
                self.data.append({"id": int(id), "text": passage})

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
        examples = [self.get_detailed_example(e['instruct'], e['query'], e['response']) for e in examples]
        self.examples_prefix = '\n\n'.join(examples) + '\n\n' 

    def get_detailed_example(self, task_description: str, query: str, response: str) -> str:
        return f'<instruct>{task_description}\n<query>{query}\n<response>{response}'

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Returns a tokenized sample."""
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
