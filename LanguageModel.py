import random
from collections import defaultdict


class BasicLanguageModel:
    def __init__(self, n_params=5):
        random.seed(42)
        self.n_params = n_params
        self.state= [{} for _ in range(n_params)]
        self.train_data, self.test_data = self.get_data()
        self.num_train_tokens = len(self.train_data)

    def get_data(self):
        pass
    def train(self):
        tokens = self.train_data
        for ind in range(1, self.n_params + 1):
            counts = self.state[ind - 1]
            for i in range(len(tokens) - ind + 1):
                context = tuple(tokens[i:i + ind -1])
                next_token = tokens[i+ind-1]
                if context not in counts:
                    counts[context] = defaultdict(int)
                counts[context][next_token] = counts[context][next_token] + 1


    def predict_next_token(self, context):
        for n in range(self.n_params, 1, -1):
            if len(context) >= n-1:
                context_n = tuple(context[-(n-1):])
                counts = self.state[n-1].get(context_n)
                if counts:
                    return max(counts.items(), key=lambda x:x[1])[0]
        unigram_counts = self.state[0]. get(())
        if unigram_counts:
            return max(unigram_counts.items(), key=lambda x:x[1])[0]
        return None
        


    def generate_token(self):
        pass
    def get_probability(self):
        pass
    def compute_perplexity(self):
        pass

if __name__ == "__main__":
    model = BasicLanguageModel()
    model.train()
    def chat(message, history):
        return model.generate_text(message, 10)
    perplexity = model.compute_perplexity()