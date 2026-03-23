# Word2Vec implementation using NumPy

A pure NumPy implementation of Word2Vec with Skip-Gram and negative sampling.

## Project Structure

```
data/
    - text8             # raw corpus (download separately)
models/
    - model.npz         # saved model (created after training)
src/
    - vocabulary.py     # vocab building, subsampling, noise distribution
    - dataset.py        # training pair generation, negative sampling
    - model.py          # weight matrices, forward/backward pass
    - trainer.py        # training loop
    - embeddings.py     # similarity, analogy queries on trained embeddings
    - train.py          # entry point for training
    - test.py           # load model and query embeddings
```

## How It Works

### Architecture

Skip-Gram with negative sampling. Given a center word, the model learns to predict its surrounding context words within a sliding window.

Each word has two learned vectors: an input (center) vector stored in `W_in` and an output (context) vector stored in `W_out`. After training, `W_in` is used as the final word embeddings.

### Objective

For each (center, context) pair, the model maximises:

$$J = \log \sigma(u_o^\top v_c) + \sum_{k=1}^{K} \log \sigma(-u_k^\top v_c)$$

where $v_c$ is the center vector, $u_o$ is the true context vector, and $u_k$ are $K$ randomly sampled noise vectors. This avoids the expensive full-vocabulary softmax.

### Training Details

- **Negative sampling** — 5 noise words per positive pair, sampled from a unigram distribution raised to the 3/4 power to balance rare and frequent words
- **Subsampling** — frequent words are randomly discarded before pair generation with probability $1 - \sqrt{t / f(w)}$, where $t = 10^{-5}$
- **Random window size** — window radius sampled uniformly from $[1, max\_window]$ each step
- **Linear LR decay** — learning rate decays linearly from 0.025 to 0.0001 over the full training run

## Getting Started

### Requirements

```bash
pip install numpy
```

### Dataset

Download Text8 (first 100MB of cleaned Wikipedia):

```bash
mkdir -p data
wget http://mattmahoney.net/dc/text8.zip -P data/
unzip data/text8.zip -d data/
```

### Training

```bash
python train.py
```

### Querying Embeddings

```bash
python test.py
```
