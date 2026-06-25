"""
dataset_preprocess.py — GSCv3 PyTorch Datasets & Tokenization

Handles BPE tokenizer training, 1D Temporal Interpolation, 
and PyTorch Dataset generation for Phase 1 and Phase 2.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

# --- Configuration Defaults ---
VOCAB_SIZE = 8057
SEQ_LEN = 16
PAD_TOKEN = "<pad>"
EOP_TOKEN = "<|endofphrase|>"
EOT_TOKEN = "<|endoftext|>"

def get_or_train_tokenizer(raw_text_path="dataset_10.txt", vocab_size=VOCAB_SIZE):
    """Loads or trains a BPE tokenizer on the raw text."""
    tokenizer_path = "gsc_v3_tokenizer.json"
    
    if os.path.exists(tokenizer_path):
        return Tokenizer.from_file(tokenizer_path)
        
    print("Training new BPE Tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", PAD_TOKEN, EOP_TOKEN, EOT_TOKEN]
    )
    
    tokenizer.train([raw_text_path], trainer=trainer)
    
    # Configure post-processor to handle byte-level alignments
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")
    return tokenizer

def interpolate_phrase(tokens, seq_len):
    """
    1D Temporal Interpolation (Nearest-Neighbor).
    Stretches a sequence of tokens to exactly seq_len.
    """
    if not tokens:
        return [0] * seq_len
    return [tokens[int(i * len(tokens) / seq_len)] for i in range(seq_len)]

class Phase1Dataset(Dataset):
    """
    Dataset for Autoencoder Training.
    Yields single, interpolated phrases.
    """
    def __init__(self, json_path, tokenizer, seq_len=SEQ_LEN):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.seq_len = seq_len
        self.eop_id = tokenizer.token_to_id(EOP_TOKEN)
        self.samples = []
        
        print("Building Phase 1 Dataset...")
        for story in data:
            for phrase in story["phrases"]:
                # Tokenize and append <|endofphrase|>
                tokens = tokenizer.encode(phrase).ids + [self.eop_id]
                # Interpolate to target SEQ_LEN
                scaled_tokens = interpolate_phrase(tokens, self.seq_len)
                self.samples.append(torch.tensor(scaled_tokens, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class Phase2Dataset(Dataset):
    """
    Dataset for Latent Generator Training.
    Yields adjacent phrase pairs (current_phrase, next_phrase).
    """
    def __init__(self, json_path, tokenizer, seq_len=SEQ_LEN):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.seq_len = seq_len
        self.eop_id = tokenizer.token_to_id(EOP_TOKEN)
        self.pairs = []
        
        print("Building Phase 2 Dataset...")
        for story in data:
            phrases = story["phrases"]
            # We need at least 2 phrases to form a pair
            if len(phrases) < 2:
                continue
                
            for i in range(len(phrases) - 1):
                curr_text = phrases[i]
                next_text = phrases[i+1]
                
                curr_tokens = tokenizer.encode(curr_text).ids + [self.eop_id]
                next_tokens = tokenizer.encode(next_text).ids + [self.eop_id]
                
                curr_scaled = interpolate_phrase(curr_tokens, self.seq_len)
                next_scaled = interpolate_phrase(next_tokens, self.seq_len)
                
                self.pairs.append((
                    torch.tensor(curr_scaled, dtype=torch.long),
                    torch.tensor(next_scaled, dtype=torch.long)
                ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

if __name__ == "__main__":
    # Quick verification test
    tok = get_or_train_tokenizer()
    
    if os.path.exists("processed_dataset.json"):
        p1 = Phase1Dataset("processed_dataset.json", tok)
        p2 = Phase2Dataset("processed_dataset.json", tok)
        
        print(f"Phase 1 Samples: {len(p1)}")
        print(f"Phase 2 Pairs: {len(p2)}")
        
        if len(p2) > 0:
            c, n = p2[0]
            print("\nSample Pair 0:")
            print(f"Current Shape: {c.shape} | Next Shape: {n.shape}")
            print(f"Current Tokens: {c.tolist()}")
            print(f"Next Tokens:    {n.tolist()}")