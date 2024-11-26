import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

class DeBERTaReranker(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-large", pooling="mean", precision="fp32"):
        super(DeBERTaReranker, self).__init__()
        
        # Set torch dtype based on input parameter
        if precision == "fp32":
            self.torch_dtype = torch.float32
        elif precision == "bf16":
            self.torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Use 'fp32' or 'bf16'")
            
        self.deberta = AutoModel.from_pretrained(model_name, torch_dtype=self.torch_dtype)
        self.config = self.deberta.config
        self.fc = nn.Linear(self.deberta.config.hidden_size, 1, dtype=self.torch_dtype)
        
    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1)
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return (sum_embeddings / sum_mask).to(self.torch_dtype) 
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self.mean_pooling(last_hidden_state, attention_mask)
        score = self.fc(pooled_output)
        logits = score.squeeze(-1)
        return SequenceClassifierOutput(logits=logits)
