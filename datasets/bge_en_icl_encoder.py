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
                'query': 'what is the capital of australia',
                'response': "Canberra Canberra is the capital city of Australia. Founded following the federation of the colonies of Australia as the seat of government for the new nation, it is Australia's largest inland city and the eighth-largest city overall. Located at the northern end of the Australian Capital Territory, Canberra is an entirely planned city."},
                {'instruct': self.task,
                'query': 'who invented the world wide web',
                'response': "Tim Berners-Lee Sir Timothy John Berners-Lee, also known as TimBL, is an English engineer and computer scientist, best known as the inventor of the World Wide Web. He implemented the first successful communication between a Hypertext Transfer Protocol (HTTP) client and server via the Internet in mid-November 1989. Berners-Lee is a professor at the Massachusetts Institute of Technology (MIT) and the University of Oxford."},
                {'instruct': self.task,
                 'query': 'what is the Higgs boson',
                 'response': "Higgs Boson The Higgs boson is an elementary particle in the Standard Model of particle physics. It is the quantum excitation of the Higgs field, which is pivotal to explaining how particles acquire mass. The discovery of the Higgs boson was announced in 2012 by physicists working with the Large Hadron Collider at CERN."},
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
            texts = [self.get_detailed_instruct(text) for text in texts]
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
