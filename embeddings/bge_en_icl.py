from FlagEmbedding import FlagICLModel
from embedding_model import EmbeddingModelInterface

class BgeEnIclEmbeddingModel(EmbeddingModelInterface):
    model: FlagICLModel

    def __init__(self, use_fp16=True):
        self.model = BgeEnIclEmbeddingModel.load_model(use_fp16)

    def embedding_dim(self):
        return self.model.model.dim

    def encode_queries(self, queries):
        return self.model.encode_queries(queries)
    
    def encode_documents(self, documents):
        return self.model.encode_corpus(documents)

    @staticmethod
    def load_model(use_fp16=True):
        examples = [
        {'instruct': 'Given a query, retrieve relevant passages that answer the query.',
        'query': 'what is a virtual interface',
        'response': "A virtual interface is a software-defined abstraction that mimics the behavior and characteristics of a physical network interface. It allows multiple logical network connections to share the same physical network interface, enabling efficient utilization of network resources. Virtual interfaces are commonly used in virtualization technologies such as virtual machines and containers to provide network connectivity without requiring dedicated hardware. They facilitate flexible network configurations and help in isolating network traffic for security and management purposes."},
        {'instruct': 'Given a query, retrieve relevant passages that answer the query.',
        'query': 'causes of back pain in female for a week',
        'response': "Back pain in females lasting a week can stem from various factors. Common causes include muscle strain due to lifting heavy objects or improper posture, spinal issues like herniated discs or osteoporosis, menstrual cramps causing referred pain, urinary tract infections, or pelvic inflammatory disease. Pregnancy-related changes can also contribute. Stress and lack of physical activity may exacerbate symptoms. Proper diagnosis by a healthcare professional is crucial for effective treatment and management."}
        ]

        return FlagICLModel('BAAI/bge-en-icl', 
                            query_instruction_for_retrieval="Given a query, retrieve relevant passages that answer the query.",
                            examples_for_task=examples,
                            use_fp16=use_fp16) 