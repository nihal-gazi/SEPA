"""
config.py — Baseline LLM Configuration

Configuration matched to the ~5 Million parameter footprint of the GSC architecture
to ensure a scientifically fair comparison.
"""

import torch
import os

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Tokenizer & Sequence Specs ---
VOCAB_SIZE = 8057
MAX_SEQ_LEN = 128       # Standard LLM Context Window

# --- Model Dimensions (~5.2M Params) ---
D_MODEL = 128           
N_HEADS = 8
NUM_LAYERS = 12         # Deep enough to match GSC parameter count
DIM_FEEDFORWARD = 512

# --- Training Specs ---
BATCH_SIZE = 64
EPOCHS = 20

# --- File Paths ---
BASE_DIR = "LLM"        # Save models inside the LLM folder
DATASET_PATH = "dataset_10.txt"