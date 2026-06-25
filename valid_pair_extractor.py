"""
extract_valid_pairs.py — GSCv3 Valid Codebook Mapper

Passes the training dataset through the frozen Phase 1 Autoencoder 
to record all valid (Stage 1, Stage 2) codebook combinations.
This dictionary is used to mask Out-Of-Distribution guesses during inference.
"""

import os
import json
import torch
from torch.utils.data import DataLoader

import config
from dataset_preprocess import get_or_train_tokenizer, Phase1Dataset
from models import Observer, ResidualVectorQuantizer

def main():
    device = config.DEVICE
    print(f"Using device: {device}")
    
    tokenizer = get_or_train_tokenizer(raw_text_path="dataset_10.txt", vocab_size=config.VOCAB_SIZE)
    dataset = Phase1Dataset(config.DATASET_PATH, tokenizer, seq_len=config.SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print("Loading Frozen Phase 1 Models...")
    observer = Observer(
        vocab_size=config.VOCAB_SIZE, d_model=config.D_MODEL, 
        seq_len=config.SEQ_LEN, latent_len=config.LATENT_LEN, 
        nhead=config.N_HEADS, num_layers=6
    ).to(device)
    
    quantizer = ResidualVectorQuantizer(
        d_model=config.D_MODEL, num_quantizers=config.NUM_QUANTIZERS, 
        codebook_size=config.CODEBOOK_SIZE, codebook_dim=config.CODEBOOK_DIM
    ).to(device)
    
    observer.load_state_dict(torch.load(os.path.join(config.BASE_DIR, "observer.pt"), map_location=device, weights_only=True))
    quantizer.load_state_dict(torch.load(os.path.join(config.BASE_DIR, "quantizer.pt"), map_location=device, weights_only=True))
    
    observer.eval()
    quantizer.eval()
    
    valid_pairs = {}
    total_pairs_found = 0
    
    print("Extracting valid codebook combinations...")
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            z = observer(batch)
            _, indices, _ = quantizer(z) # (B, LATENT_LEN, 2)
            
            # Flatten to a list of (S1, S2) pairs
            s1_s2_pairs = indices.view(-1, 2).cpu().numpy()
            
            for s1, s2 in s1_s2_pairs:
                s1, s2 = int(s1), int(s2)
                if s1 not in valid_pairs:
                    valid_pairs[s1] = set()
                if s2 not in valid_pairs[s1]:
                    valid_pairs[s1].add(s2)
                    total_pairs_found += 1
                    
    # Convert sets to lists for JSON serialization
    valid_pairs_serializable = {k: list(v) for k, v in valid_pairs.items()}
    
    output_path = os.path.join(config.BASE_DIR, "valid_pairs.json")
    with open(output_path, "w") as f:
        json.dump(valid_pairs_serializable, f)
        
    print(f"Extraction Complete!")
    print(f"Found {total_pairs_found} unique combinations out of {config.CODEBOOK_SIZE * config.CODEBOOK_SIZE} possible.")
    print(f"Saved dictionary to '{output_path}'.")

if __name__ == "__main__":
    main()