#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train sub‑token embeddings for a Polish corpus from scratch (no pre‑trained vectors).

Pipeline:
1. Train a SentencePiece BPE tokenizer.
2. Build a sparse word–context co‑occurrence matrix with a symmetric window.
3. Convert the matrix to Positive PMI (PPMI).
4. Reduce dimensionality with Truncated SVD → dense embeddings.

Outputs:
    * pl_bpe.model / pl_bpe.vocab – the trained tokenizer
    * embeddings.npy          – NumPy array of shape (vocab_size, dims)
"""

from pathlib import Path
from collections import Counter
from typing import List, Tuple

import numpy as np
import sentencepiece as spm
from scipy import sparse
from tqdm import tqdm
from numpy.linalg import norm
from sklearn.decomposition import TruncatedSVD

# ---------------------------------------------------------------------------
# Parameters (replace CLI)
# ---------------------------------------------------------------------------
CORPUS_PATH = Path("Example.txt")
VOCAB_SIZE = 10_000
WINDOW_SIZE = 4
EMBED_DIM = 300
MODEL_PREFIX = "pl_bpe"

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def train_tokenizer(corpus_path: Path, vocab_size: int, model_prefix: str) -> spm.SentencePieceProcessor:
    """Train a SentencePiece BPE tokenizer and return a processor."""
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9995,
        model_type="bpe",
    )
    return spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

# ---------------------------------------------------------------------------
# Co‑occurrence helpers
# ---------------------------------------------------------------------------

def window_pairs(ids: List[int], window: int):
    """Yield (center, context) id pairs within a symmetrical window."""
    for i, center in enumerate(ids):
        l = max(0, i - window)
        r = min(len(ids), i + window + 1)
        for ctx in ids[l:i] + ids[i + 1 : r]:
            yield center, ctx

def build_cooc_matrix(corpus_path: Path, sp: spm.SentencePieceProcessor, window: int):
    """Build a symmetric sparse co‑occurrence matrix and gather basic stats."""
    vocab_size = sp.get_piece_size()
    rows, cols = [], []
    token_freq = Counter()
    sent_lens = []

    with corpus_path.open("r", encoding="utf‑8") as f:
        for line in tqdm(f, desc="Encoding & collecting"):
            ids = sp.encode(line.strip().lower(), out_type=int)
            if not ids:
                continue
            token_freq.update(ids)
            sent_lens.append(len(ids))
            for i, j in window_pairs(ids, window):
                rows.append(i)
                cols.append(j)

    data = np.ones(len(rows), dtype=np.float32)
    cooc = sparse.coo_matrix((data, (rows, cols)), shape=(vocab_size, vocab_size), dtype=np.float32)
    return (cooc + cooc.T).tocsr(), token_freq, sent_lens

# ---------------------------------------------------------------------------
# PPMI & SVD
# ---------------------------------------------------------------------------

def ppmi_transform(cooc: sparse.csr_matrix) -> sparse.csr_matrix:
    """Convert raw counts to Positive PMI."""
    S = cooc.sum()
    row_sums = np.asarray(cooc.sum(axis=1)).flatten()
    col_sums = np.asarray(cooc.sum(axis=0)).flatten()
    row, col = cooc.nonzero()

    p_ij = cooc.data / S
    p_i = row_sums[row] / S
    p_j = col_sums[col] / S

    pmi = np.log2(p_ij / (p_i * p_j))
    ppmi_vals = np.maximum(pmi, 0.0)

    return sparse.coo_matrix((ppmi_vals, (row, col)), shape=cooc.shape).tocsr()

def svd_embeddings(M: sparse.csr_matrix, dims: int, random_state: int = 0) -> np.ndarray:
    """Return a dense embedding matrix via Truncated SVD."""
    svd = TruncatedSVD(n_components=dims, random_state=random_state)
    return svd.fit_transform(M)

# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class SubtokenEmbeddings:
    def __init__(self, sp: spm.SentencePieceProcessor, emb: np.ndarray):
        self.sp = sp
        self.emb = emb

    def nearest(self, token: str, k: int = 10) -> List[Tuple[str, float]]:
        tid = self.sp.piece_to_id(token)
        if tid < 0:
            raise ValueError(f"Token '{token}' not in vocab.")
        v = self.emb[tid]
        sims = self.emb @ v / (norm(self.emb, axis=1) * norm(v) + 1e-9)
        best = sims.argsort()[-k-1:][::-1][1:]
        return [(self.sp.id_to_piece(int(i)), float(sims[i])) for i in best]

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("▶️  Training tokenizer …")
    sp = train_tokenizer(CORPUS_PATH, VOCAB_SIZE, MODEL_PREFIX)
    print(f"✔️  Vocab size: {sp.get_piece_size()}")

    print("▶️  Building co‑occurrence matrix …")
    cooc, token_freq, sent_lens = build_cooc_matrix(CORPUS_PATH, sp, WINDOW_SIZE)

    print("▶️  Applying PPMI …")
    ppmi = ppmi_transform(cooc)

    print("▶️  Running SVD …")
    emb = svd_embeddings(ppmi, EMBED_DIM)

    np.save("embeddings.npy", emb)
    print("✔️  Saved embeddings → embeddings.npy")

    # Demo
    se = SubtokenEmbeddings(sp, emb)
    for demo in ("Polska", "warszawa", "nauka"):
        try:
            print(f"Nearest to '{demo}':", se.nearest(demo))
        except ValueError:
            pass
