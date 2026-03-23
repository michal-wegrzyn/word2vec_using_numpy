from vocabulary import Vocabulary
import numpy as np


class EmbeddingSpace:
    def __init__(self, embeddings: np.ndarray, vocab: Vocabulary):
        self.embeddings = embeddings.copy()
        self.vocab = vocab

        # Normalize W_in rows to unit length on init — cosine similarity then becomes just a dot product, which is fast.
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)

    def vector(self, word: str) -> np.ndarray:
        """Get the embedding vector for a given word."""
        if word not in self.vocab.word_to_idx:
            raise KeyError(f"'{word}' not in vocabulary")
        return self.embeddings[self.vocab.word_to_idx[word]]

    def most_similar(self, word: str, top_n: int = 10) -> list[tuple[str, float]]:
        word_vec = self.vector(word)
        similarities = np.dot(self.embeddings, word_vec)
        top_indices = np.argpartition(similarities, -top_n)[-top_n:]
        return sorted(
            [(self.vocab.idx_to_word[idx], similarities[idx]) for idx in top_indices],
            key=lambda x: x[1],
            reverse=True,
        )

    def analogy(
        self, positive: list[str], negative: list[str], top_n: int = 10
    ) -> list[tuple[str, float]]:
        v = sum(self.vector(w) for w in positive) - sum(
            self.vector(w) for w in negative
        )
        v /= np.linalg.norm(v)  # Normalize to unit length
        similarities = np.dot(self.embeddings, v)
        top_indices = np.argpartition(similarities, -top_n)[-top_n:]
        return sorted(
            [(self.vocab.idx_to_word[idx], similarities[idx]) for idx in top_indices],
            key=lambda x: x[1],
            reverse=True,
        )

    def similarity(self, word_a: str, word_b: str) -> float:
        vec_a = self.vector(word_a)
        vec_b = self.vector(word_b)
        return np.dot(vec_a, vec_b)
