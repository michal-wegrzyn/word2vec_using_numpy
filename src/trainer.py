from model import Word2Vec
from dataset import TrainingDataset, NoiseSampler
import numpy as np


class Trainer:
    def __init__(
        self,
        model: Word2Vec,
        dataset: TrainingDataset,
        noise_sampler: NoiseSampler,
        learning_rate: float = 0.025,
        neg_samples: int = 5,
    ):
        self.model = model
        self.dataset = dataset
        self.noise_sampler = noise_sampler
        self.starting_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.neg_samples = neg_samples

    def _update(self, grads: dict) -> None:
        np.add.at(
            self.model.W_in,
            grads["centers"][0],
            -self.learning_rate * grads["centers"][1],
        )
        np.add.at(
            self.model.W_out,
            grads["contexts"][0],
            -self.learning_rate * grads["contexts"][1],
        )
        np.add.at(
            self.model.W_out,
            grads["negatives"][0],
            -self.learning_rate * grads["negatives"][1],
        )

    def _linear_lr_decay(self, step: int, total_steps: int) -> float:
        if self.starting_learning_rate <= 0.0001:
            return self.starting_learning_rate
        assert step <= total_steps, "Step cannot exceed total steps for decay schedule."
        return 0.0001 + (self.starting_learning_rate - 0.0001) * (
            1 - (step - 1) / (total_steps - 1)
        )

    def train(
        self,
        encoded: list[int],
        epochs: int = 5,
        batch_size: int = 256,
        log_every: int = 10_000,
    ) -> list[float]:
        loss_history = []
        global_step = 0
        for epoch in range(1, epochs + 1):
            max_steps = global_step + (epochs - epoch + 1) * (
                (self.dataset.max_generate_pairs_length(encoded) - 1) // batch_size + 1
            )
            print(f"Epoch {epoch}/{epochs}")
            running_loss = 0.0
            total_running_loss = 0.0
            step = 0
            for centers, contexts in self.dataset.batch_iterator(encoded, batch_size):
                negatives = np.array(
                    [
                        self.noise_sampler.sample(self.neg_samples, exclude=[ctx])
                        for ctx in contexts
                    ]
                )
                loss, cache = self.model.forward(centers, contexts, negatives)
                grads = self.model.backward(cache)
                self.learning_rate = self._linear_lr_decay(global_step, max_steps)
                self._update(grads)
                running_loss += loss
                total_running_loss += loss
                step += 1
                global_step += 1
                if step % log_every == 0:
                    avg_loss = running_loss / log_every
                    print(
                        f"Step {step}, Avg Loss: {avg_loss:.4f}, LR: {self.learning_rate:.6f}"
                    )
                    loss_history.append(avg_loss)
                    running_loss = 0.0

            if step % log_every:
                avg_loss = running_loss / (step % log_every)
                print(
                    f"Step {step}, Avg Loss: {avg_loss:.4f}, LR: {self.learning_rate:.6f}"
                )
                loss_history.append(avg_loss)
            print(
                f"Epoch {epoch} completed. Avg Loss: {total_running_loss/step:.4f}, LR: {self.learning_rate:.6f}"
            )
        return loss_history
