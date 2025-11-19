"""
This code trains a SentencePiece tokenizer, uses it to stream tokens,
and then trains a Word2Vec model (a simple neural network approach)
to generate sub-token embeddings, replacing the old PPMI+SVD method.
"""

from pathlib import Path
from typing import List, Generator, Iterable
import numpy as np
import sentencepiece as spm
import logging
from gensim.models import Word2Vec
from tqdm import tqdm

# Set up logging for Gensim to see progress
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Parameters
CORPUS_PATH = Path("AllDatasets.txt")   # dataset to use
VOCAB_SIZE = 10_000                    # Sub‑token vocabulary size
WINDOW_SIZE = 16                    # Context window (in size of tokens)
EMBED_DIM = 90                      # dimensionality
MODEL_PREFIX = "pl_bpe_"        # Prefix for *.model / *.vocab files
MIN_TOKEN_COUNT = 5                 # Ignore tokens that appear less than this

#  Step 1: Tokenizer
def train_tokenizer(corpus_path: Path, vocab_size: int, model_prefix: str) -> spm.SentencePieceProcessor:
    """Train a SentencePiece BPE tokenizer"""
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="bpe",
        # Use a large max_sentence_length to avoid splitting long lines if needed
        # max_sentence_length=100000 
    )
    processor = spm.SentencePieceProcessor()
    processor.load(f"{model_prefix}.model")
    return processor

# --- Step 2: Corpus Reader (The memory-efficient streaming replacement) ---
class SentencePieceCorpusReader:
    """
    An iterator that reads the corpus file line-by-line, tokenizes each line
    using the SentencePiece model, and yields the list of sub-tokens (strings).
    
    This is memory efficient, as only one line is processed at a time, which
    is exactly what Gensim's Word2Vec needs for training large corpora.
    """
    def __init__(self, corpus_path: Path, sp: spm.SentencePieceProcessor):
        self.corpus_path = corpus_path
        self.sp = sp
        # Estimate total lines for tqdm progress bar
        try:
            with self.corpus_path.open('r', encoding='utf-8') as f:
                self.total_lines = sum(1 for _ in f)
        except Exception:
            self.total_lines = None
            
    def __iter__(self) -> Generator[List[str], None, None]:
        """Iterate over the corpus, yielding tokenized sentences."""
        
        # Use tqdm to show progress (since Gensim's internal progress is verbose)
        with self.corpus_path.open("r", encoding="utf-8") as f:
            for line in tqdm(f, total=self.total_lines, desc="Streaming tokens to Word2Vec"):
                # Use out_type=str to get the actual sub-tokens for Gensim
                tokens = self.sp.encode(line.strip().lower(), out_type=str)
                if tokens:
                    yield tokens

# --- Main Execution ---
if __name__ == "__main__":
    print("▶️  Training tokenizer (SentencePiece) …")
    sp = train_tokenizer(CORPUS_PATH, VOCAB_SIZE, MODEL_PREFIX)
    # The actual vocabulary size might be slightly smaller or larger than VOCAB_SIZE
    print(f"✔️  Tokenizer trained and loaded. Actual vocab size: {sp.get_piece_size()}")

    # Initialize the streaming reader
    sentences_stream = SentencePieceCorpusReader(CORPUS_PATH, sp)

    print("▶️  Training Word2Vec model (NN-based embeddings) …")
    
    # --- Step 3: Train Word2Vec (Replacing Co-occurrence/PPMI/SVD) ---
    # Gensim handles the neural network magic efficiently
    w2v_model = Word2Vec(
        # The CorpusReader handles streaming the data (input)
        sentences=sentences_stream,
        
        # Hyperparameters matching your original parameters
        vector_size=EMBED_DIM,        # vector_size is EMBED_DIM (e.g., 30)
        window=WINDOW_SIZE,           # window is the context size (e.g., 12)
        
        # Standard Word2Vec parameters
        min_count=MIN_TOKEN_COUNT,    # Ignore low-frequency tokens
        sg=1,                         # 1 for Skip-gram (better for rare words), 0 for CBOW
        workers=8,                    # Use multiple CPU cores for speed (adjust as needed)
        epochs=15                     # Number of training iterations over the data
    )

    print("✔️  Word2Vec training complete.")

    # --- Step 4: Extract and Save Embeddings ---
    # Gensim stores the final vectors in the .wv (Word Vectors) attribute
    # Since SentencePiece IDs map directly to the vector matrix rows, 
    # we can construct the final embedding matrix using the entire Word2Vec KeyedVectors.
    
    # Get the embedding matrix (NumPy array)
    # Note: Gensim only stores vectors for words that met the min_count threshold.
    # To match the vocab size structure, we'll create a full matrix and fill it.
    
    full_vocab_size = sp.get_piece_size()
    embedding_matrix = np.zeros((full_vocab_size, EMBED_DIM), dtype=np.float32)

    # Fill the matrix. Tokens not in w2v_model.wv will remain zero (the default)
    # This loop ensures the resulting array shape matches the tokenizer's full vocab.
    
    for i in range(full_vocab_size):
        piece = sp.id_to_piece(i)
        # Check if the token was trained by Word2Vec (met min_count)
        if piece in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[piece]
            
    # Save the final matrix
    np.save("embeddings.npy", embedding_matrix)
    print("✔️  Saved new Word2Vec embeddings → embeddings_w2v.npy")
    print("✅ Training and embedding generation complete using Gensim.")
    
