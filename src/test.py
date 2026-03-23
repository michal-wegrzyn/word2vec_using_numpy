from vocabulary import Vocabulary
from model import Word2Vec
from embeddings import EmbeddingSpace

with open("../datasets/text8", "r", encoding="utf-8") as f:
    words = f.read().split()[:1000000]

vocab = Vocabulary(words, min_count=5)
print(f"Vocabulary size: {vocab.size}")

model = Word2Vec.load("../models/model.npz")
embedding_space = EmbeddingSpace(model.get_embeddings(), vocab)

print(embedding_space.most_similar("king"))
print(embedding_space.most_similar("vector"))
print(embedding_space.most_similar("day"))
print(embedding_space.most_similar("italy"))

print(embedding_space.analogy(positive=["king", "woman"], negative=["man"]))
print(embedding_space.analogy(positive=["rome", "france"], negative=["italy"]))
