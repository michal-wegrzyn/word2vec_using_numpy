from collections import Counter
import math
import numpy as np


class Vocabulary:
    def __init__(self, words: list[str], min_count: int = 5, subsample_t: float = 1e-5):
        self.min_count = min_count
        self.subsample_t = subsample_t
        words_counter = Counter(words)
        self.words = [w for w, c in words_counter.items() if c >= min_count]
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = {}
        for idx, word in enumerate(self.words):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
            self.word_freq[word] = words_counter[word]
        self.size = len(self.words)
        self.total_count = sum(self.word_freq.values())
        self.subsample_probability = {
            w: max(
                0.0,
                1 - math.sqrt(self.subsample_t / (words_counter[w] / self.total_count)),
            )
            for w in self.words
        }
        freqs = np.array([self.word_freq[w] for w in self.words], dtype=np.float64)
        freqs_powered = freqs**0.75
        self.noise_distribution = freqs_powered / freqs_powered.sum()

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.word_to_idx[word] for word in tokens if word in self.word_to_idx]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.idx_to_word[idx] for idx in indices]
