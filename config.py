"""
config.py — GSCv3 Global Configuration

Centralized hyperparameters for the General Compression System (GSCv3).
Configured for ~5M parameters with a 2x compression bottleneck.
"""

import torch

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Tokenizer & Sequence Specs ---
VOCAB_SIZE = 8057
SEQ_LEN = 32            # Updated: The interpolated, padded text length
LATENT_LEN = 16         # Updated: The compressed bottleneck length (2x compression)

# --- Model Dimensions ---
D_MODEL = 128           # Core embedding and transformer dimension
N_HEADS = 8

# --- Vector Quantization (RVQ) Specs ---
NUM_QUANTIZERS = 2      # Two independent codebook stages
CODEBOOK_SIZE = 2048    # Size of each codebook
CODEBOOK_DIM = 32       # Dimension of the discrete vectors (FSQ/RVQ)
EMA_DECAY = 0.99        # Decay rate for codebook updates

# --- Training Specs ---
BATCH_SIZE = 256
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 20

# --- File Paths ---
BASE_DIR = "."          # Directory to save/load models
DATASET_PATH = "processed_dataset.json"