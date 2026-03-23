import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class Word2Vec:
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.W_in = np.random.uniform(
            low=-0.5 / embedding_dim,
            high=0.5 / embedding_dim,
            size=(vocab_size, embedding_dim),
        )
        self.W_out = np.random.uniform(
            low=-0.5 / embedding_dim,
            high=0.5 / embedding_dim,
            size=(vocab_size, embedding_dim),
        )

    def forward(
        self, centers: np.ndarray, contexts: np.ndarray, negatives: np.ndarray
    ) -> tuple[float, dict]:
        V_c = self.W_in[centers]  # shape: (B, d)
        U_pos = self.W_out[contexts]  # shape: (B, d)
        U_neg = self.W_out[negatives]  # shape: (B, K, d)

        score_pos = np.sum(U_pos * V_c, axis=1)  # shape: (B,)

        scores_neg = np.einsum("bd,bkd->bk", V_c, U_neg)  # shape: (B, K)

        loss = -np.mean(
            np.log(sigmoid(score_pos)) + np.sum(np.log(sigmoid(-scores_neg)), axis=1)
        )

        cache = {
            "centers": centers,
            "contexts": contexts,
            "negatives": negatives,
            "V_c": V_c,
            "U_pos": U_pos,
            "U_neg": U_neg,
            "score_pos": score_pos,
            "scores_neg": scores_neg,
        }
        return loss, cache

    def backward(self, cache: dict) -> dict:
        d_pos = (sigmoid(cache["score_pos"]) - 1)[:, None]  # shape: (B, 1)
        d_neg = sigmoid(cache["scores_neg"])  # shape: (B, K)

        d_V_c = d_pos * cache["U_pos"] + np.einsum(  # shape: (B, d)
            "bk,bkd->bd", d_neg, cache["U_neg"]
        )

        d_U_pos = d_pos * cache["V_c"]  # shape: (B, d)

        d_U_neg = d_neg[:, :, None] * cache["V_c"][:, None, :]  # shape: (B, K, d)

        return {
            "centers": (cache["centers"], d_V_c),
            "contexts": (cache["contexts"], d_U_pos),
            "negatives": (cache["negatives"], d_U_neg),
        }

    def get_embeddings(self) -> np.ndarray:
        return self.W_in

    def save(self, path: str) -> None:
        np.savez(path, W_in=self.W_in, W_out=self.W_out)

    @classmethod
    def load(cls, path: str) -> "Word2Vec":
        data = np.load(path)
        vocab_size, embedding_dim = data["W_in"].shape
        model = cls(vocab_size, embedding_dim)
        model.W_in = data["W_in"]
        model.W_out = data["W_out"]
        return model
