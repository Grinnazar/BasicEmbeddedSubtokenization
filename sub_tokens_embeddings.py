"""
This code trains a SentencePiece tokenizer, builds a co-occurrence matrix,
applies PPMI transformation, performs Truncated SVD, and saves the
resulting sub-token embeddings.
"""

from pathlib import Path
from typing import List 
import numpy as np
import sentencepiece as spm
from scipy import sparse
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD

# Parameters
CORPUS_PATH = Path("AllDatasets.txt")   # dataset to use
VOCAB_SIZE = 50000                    # Sub‑token vocabulary size
WINDOW_SIZE = 7                     # Context window (in size of tokens)
EMBED_DIM = 45                      # dimensionality
MODEL_PREFIX = "pl_bpe_"        # Prefix for *.model / *.vocab files

# Tokeniser (DS to tokens)
def train_tokenizer(corpus_path: Path, vocab_size: int, model_prefix: str) -> spm.SentencePieceProcessor:
    """Train a SentencePiece BPE tokenizer"""
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
    )
    processor = spm.SentencePieceProcessor() # Changed to initialize and then load
    processor.load(f"{model_prefix}.model")  # Explicitly load the model after training
    return processor

# Co‑occurrence helpers
def window_pairs(ids: List[int], window: int):
    """Yield (center, context) id pairs within a symmetrical window."""
    for i, center in enumerate(ids):
        l = max(0, i - window)
        r = min(len(ids), i + window + 1)
        # Iterate through context tokens to the left (ids[l:i])
        # and to the right (ids[i+1:r])
        for ctx_token_id in ids[l:i] + ids[i+1:r]:
            yield center, ctx_token_id

def build_cooc_matrix_optimized(corpus_path: Path, sp: spm.SentencePieceProcessor, window: int, chunk_size_lines: int = 10000):
    """Build a symmetric sparse co‑occurrence matrix, optimized for memory.

    Processes the corpus in chunks to avoid storing all pairs in memory at once.
    """
    vocab_size = sp.get_piece_size()
    main_cooc_accumulator = sparse.csr_matrix((vocab_size, vocab_size), dtype=np.float32)

    # Temporary lists for collecting pairs within a chunk
    rows_chunk, cols_chunk = [], []
    lines_processed_in_chunk = 0

    print(f"Building co-occurrence matrix in chunks of {chunk_size_lines} lines...")
    with corpus_path.open("r", encoding="utf‑8") as f:
        for line in tqdm(f, desc="Encoding & collecting"):
            ids = sp.encode(line.strip().lower(), out_type=int)
            if not ids:
                continue

            for i_token_id, j_token_id in window_pairs(ids, window):
                rows_chunk.append(i_token_id)
                cols_chunk.append(j_token_id)

            lines_processed_in_chunk += 1

            if lines_processed_in_chunk >= chunk_size_lines:
                if rows_chunk: # Ensure there's something to add
                    data_chunk = np.ones(len(rows_chunk), dtype=np.float32)
                    chunk_cooc = sparse.coo_matrix((data_chunk, (rows_chunk, cols_chunk)),
                                                   shape=(vocab_size, vocab_size), dtype=np.float32)
                    main_cooc_accumulator += chunk_cooc # Add chunk's matrix to the main accumulator
                    rows_chunk, cols_chunk = [], []
                lines_processed_in_chunk = 0

        # Process any remaining pairs after the loop
        if rows_chunk:
            data_chunk = np.ones(len(rows_chunk), dtype=np.float32)
            chunk_cooc = sparse.coo_matrix((data_chunk, (rows_chunk, cols_chunk)),
                                           shape=(vocab_size, vocab_size), dtype=np.float32)
            main_cooc_accumulator += chunk_cooc

    return main_cooc_accumulator.tocsr()


# PPMI & SVD
def ppmi_transform(cooc: sparse.csr_matrix) -> sparse.csr_matrix:
    """Convert raw counts to Positive PMI."""
    S = cooc.sum()
    if S == 0: # Avoid division by zero if matrix is empty
        return sparse.csr_matrix(cooc.shape, dtype=np.float32)

    row_sums = np.asarray(cooc.sum(axis=1)).flatten()
    col_sums = np.asarray(cooc.sum(axis=0)).flatten()

    # Ensure no zero sums in denominators for p_i, p_j to prevent division by zero/warnings
    # Add a small epsilon where sums are zero.
    row_sums_nz = row_sums + 1e-9 * (row_sums == 0)
    col_sums_nz = col_sums + 1e-9 * (col_sums == 0)

    row, col = cooc.nonzero()

    p_ij = cooc.data / S
    p_i = row_sums_nz[row] / S
    p_j = col_sums_nz[col] / S

    denominator = p_i * p_j
    pmi_arg = np.zeros_like(p_ij)
    valid_mask = denominator > 1e-12
    pmi_arg[valid_mask] = p_ij[valid_mask] / denominator[valid_mask]

    # Calculate log2 only for positive arguments
    log_pmi_arg = np.zeros_like(pmi_arg)
    positive_mask = pmi_arg > 0
    log_pmi_arg[positive_mask] = np.log2(pmi_arg[positive_mask])

    ppmi_vals = np.maximum(log_pmi_arg, 0.0) # PMI values are log_pmi_arg, so take max(0, log_pmi_arg)

    return sparse.coo_matrix((ppmi_vals, (row, col)), shape=cooc.shape).tocsr()

def svd_embeddings(M: sparse.csr_matrix, dims: int, random_state: int = 0) -> np.ndarray:
    """Return a dense embedding matrix via Truncated SVD."""
    svd = TruncatedSVD(n_components=dims, random_state=random_state)
    return svd.fit_transform(M)


print("▶️  Training tokenizer …")

sp = train_tokenizer(CORPUS_PATH, VOCAB_SIZE, MODEL_PREFIX)
print(f"✔️  Tokenizer trained and loaded. Vocab size: {sp.get_piece_size()}") # Adjusted message

print("▶️  Building co‑occurrence matrix …")
cooc = build_cooc_matrix_optimized(CORPUS_PATH, sp, WINDOW_SIZE, chunk_size_lines=35_000)

print("▶️  Applying PPMI …")
ppmi = ppmi_transform(cooc)

print("▶️  Running SVD …")
emb = svd_embeddings(ppmi, EMBED_DIM)

np.save("embeddings.npy", emb)
print("✔️  Saved embeddings → embeddings.npy")
print("✅ Training and embedding generation complete.")
