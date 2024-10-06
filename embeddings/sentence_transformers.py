import torch
from sentence_transformers import SentenceTransformer
from embeddings.embedding_model import EmbeddingModelInterface

class SentenceTransformersEmbeddingModel(EmbeddingModelInterface):
    model: SentenceTransformer

    def __init__(self, model_name, max_seq_len=None):
        self.model = SentenceTransformersEmbeddingModel.get_embedding_model(model_name, max_seq_len)

    def embedding_dim(self):
        return self.model.get_sentence_embedding_dimension()
    
    def encode_queries(self, queries):
        return self.model.encode(queries)
    
    def encode_documents(self, documents):
        return self.model.encode(documents)

    @staticmethod
    def get_embedding_model(model_name, max_seq_len=None):
        model = SentenceTransformer(
            model_name, 
            trust_remote_code=True, 
            device="cuda" if torch.cuda.is_available() else "cpu",
            # model_kwargs={"torch_dtype": torch.float16}
        )
        if max_seq_len:
            model.max_seq_length = max_seq_len

        return model