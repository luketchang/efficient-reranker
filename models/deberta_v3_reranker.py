import torch
import torch.nn as nn
from transformers import DebertaModel

class DeBERTaReranker(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-large", pooling="mean"):
        super(DeBERTaReranker, self).__init__()
        self.deberta = DebertaModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.deberta.config.hidden_size, 1)
        
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).float()  # Shape: (batch_size, seq_len, 1)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask
        
    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.mean_pooling(last_hidden_state, attention_mask)
        score = self.fc(pooled_output)
        return score.squeeze(-1)
