import torch
from sentence_transformers import SentenceTransformer

def get_embedding_model(model_name, max_seq_len=None):
    """
    Load the sentence transformer model and adjust the max_seq_length if provided.
    """
    model = SentenceTransformer(
        model_name, 
        trust_remote_code=True, 
        device="cuda" if torch.cuda.is_available() else "cpu",
        # model_kwargs={"torch_dtype": torch.float16}
    )
    if max_seq_len:
        model.max_seq_length = max_seq_len
    embedding_dim = model.get_sentence_embedding_dimension()
    return model, embedding_dim