import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosime_similarity

books = [{'title':"", "content":""},{'title':"", "content":""},{'title':"", "content":""}]

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [f'{book['title']}.{book["content"]}' for book in books]
embeddings = model.encode(texts)
book_id = 0
similarities = cosime_similarity([embeddings[book_id]], embeddings)[0]
similar_indices = np.argsort(similarities)[::-1][1:3]
recommendations = [books[idx] for idx in similar_indices]