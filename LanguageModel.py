import random

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
        pass
    def predict_next_token(self):
        pass
    def generate_token(self):
        pass
    def get_probability(self):
        pass
    def compute_perplexity(self):
        pass
