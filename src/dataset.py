import numpy as np
from vocabulary import Vocabulary
import random
from typing import Iterator


class NoiseSampler:
    def __init__(self, noise_distribution: np.ndarray):
        self.noise_distribution = noise_distribution

    def sample(self, k: int, exclude: list[int]) -> np.ndarray:
        """Draw k negative sample indices from the noise distribution.
        Exclude the positive context word to avoid sampling it as negative.
        """
        exclude_set = set(exclude)
        samples = []
        while len(samples) < k:
            candidates = np.random.choice(
                len(self.noise_distribution),
                size=k,  # draw k at once, likely enough
                p=self.noise_distribution,
            )
            for c in candidates:
                if c not in exclude_set and len(samples) < k:
                    samples.append(c)
        return np.array(samples)


class TrainingDataset:
    def __init__(self, vocab: Vocabulary, max_window_size: int = 5):
        self.vocab = vocab
        self.max_window_size = max_window_size

    def subsample(self, encoded: list[int]) -> list[int]:
        """Randomly drop frequent words according to their subsample probability."""
        return [
            idx
            for idx in encoded
            if random.random()
            < self.vocab.subsample_probability[self.vocab.idx_to_word[idx]]
        ]

    def generate_pairs(self, encoded: list[int]) -> Iterator[tuple[int, int]]:
        """Generate (center, context) pairs from the encoded text."""
        for i, center in enumerate(encoded):
            window_size = random.randint(1, self.max_window_size)
            context_indices = list(range(max(0, i - window_size), i)) + list(
                range(i + 1, min(len(encoded), i + window_size + 1))
            )
            for j in context_indices:
                yield center, encoded[j]

    def max_generate_pairs_length(self, encoded: list[int]) -> int:
        return 2 * self.max_window_size * len(encoded)

    def batch_iterator(
        self, encoded: list[int], batch_size: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate batches of (center, context) pairs."""
        batch_centers, batch_contexts = [], []
        for center, context in self.generate_pairs(encoded):
            batch_centers.append(center)
            batch_contexts.append(context)
            if len(batch_centers) == batch_size:
                yield np.array(batch_centers), np.array(batch_contexts)
                batch_centers, batch_contexts = [], []

        # The last, partial batch
        if batch_centers:
            yield np.array(batch_centers), np.array(batch_contexts)
