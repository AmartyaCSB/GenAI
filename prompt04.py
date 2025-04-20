from sentence_transformers import SentenceTransformer, util
from llama_cpp import ChatGrog  # Assuming a wrapper for llama-cpp

class PromptEval:
    def __init__(self):
        self.llm = ChatGrog(model="llama-3.1-8b-instant")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

    def compute_specificity(self, response):
        words = response.split()
        unique_words = set(words)
        return len(unique_words) / len(words) if words else 0

    def compute_relevance(self, response, reference="quantum mechanics"):
        # Compute embeddings
        response_embedding = self.sentence_model.encode(response, convert_to_tensor=True)
        reference_embedding = self.sentence_model.encode(reference, convert_to_tensor=True)

        # Cosine similarity as a proxy for relevance
        similarity_score = util.pytorch_cos_sim(response_embedding, reference_embedding).item()
        return similarity_score


if __name__ == "__main__":
    eval = PromptEval()

    joey_speech = """It's a love based on giving and receiving, as well as having and sharing..."""
    quantum_mechanics = """Quantum mechanics reveals a universe fundamentally different..."""

    joey_specificity = eval.compute_specificity(joey_speech)
    quantum_specificity = eval.compute_specificity(quantum_mechanics)

    joey_relevance = eval.compute_relevance(joey_speech, reference="philosophy of love")
    quantum_relevance = eval.compute_relevance(quantum_mechanics, reference="quantum mechanics")

    print("Joey Specificity:", joey_specificity)
    print("Quantum Specificity:", quantum_specificity)
    print("Joey Relevance:", joey_relevance)
    print("Quantum Relevance:", quantum_relevance)
