from abc import ABC, abstractmethod

class EmbeddingModelInterface(ABC):
    @abstractmethod
    def embedding_dim(self):
        pass
    
    @abstractmethod
    def encode_queries(self, queries):
        pass

    @abstractmethod
    def encode_documents(self, documents):
        pass
