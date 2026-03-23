from vocabulary import Vocabulary
from model import Word2Vec
from dataset import TrainingDataset, NoiseSampler
from trainer import Trainer
from embeddings import EmbeddingSpace
import os

with open("../datasets/text8", "r", encoding="utf-8") as f:
    # Use only the first 1 million words for faster training
    words = f.read().split()[:1000000]

vocab = Vocabulary(words, min_count=5)
print(f"Vocabulary size: {vocab.size}")

noise_sampler = NoiseSampler(vocab.noise_distribution)
dataset = TrainingDataset(vocab, max_window_size=3)
model = Word2Vec(vocab.size, embedding_dim=100)
trainer = Trainer(model, dataset, noise_sampler, learning_rate=0.025, neg_samples=5)
encoded = vocab.encode(words)
trainer.train(encoded, epochs=5, batch_size=256, log_every=1000)

os.makedirs("../models", exist_ok=True)
model.save("../models/model.npz")

print("Training complete. Example similarities:")
embedding_space = EmbeddingSpace(trainer.model.get_embeddings(), vocab)

print(embedding_space.most_similar("king"))
print(embedding_space.most_similar("vector"))
print(embedding_space.most_similar("day"))
print(embedding_space.most_similar("italy"))

print(embedding_space.analogy(positive=["king", "woman"], negative=["man"]))
print(embedding_space.analogy(positive=["rome", "france"], negative=["italy"]))
