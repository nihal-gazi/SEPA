"""
train_phase2.py — GSCv3 Phase 2 Training

Loads frozen Phase 1 checkpoints, then trains the AR Latent Generator using causal sequence modeling.
"""

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import config
from dataset_preprocess import get_or_train_tokenizer, Phase2Dataset
from models import Observer, ResidualVectorQuantizer, LatentGenerator

def main():
    device = config.DEVICE
    print(f"Using device: {device}")
    
    tokenizer = get_or_train_tokenizer(raw_text_path="dataset_10.txt", vocab_size=config.VOCAB_SIZE)
    
    print("\n" + "="*50)
    print("🚀 STARTING PHASE 2: AR LATENT GENERATOR TRAINING")
    print("="*50)
    
    dataset = Phase2Dataset(config.DATASET_PATH, tokenizer, seq_len=config.SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Initialize and Load Frozen Phase 1 Models
    observer = Observer(
        vocab_size=config.VOCAB_SIZE, d_model=config.D_MODEL, 
        seq_len=config.SEQ_LEN, latent_len=config.LATENT_LEN, 
        nhead=config.N_HEADS, num_layers=6
    ).to(device)
    
    quantizer = ResidualVectorQuantizer(
        d_model=config.D_MODEL, num_quantizers=config.NUM_QUANTIZERS, 
        codebook_size=config.CODEBOOK_SIZE, codebook_dim=config.CODEBOOK_DIM
    ).to(device)
    
    # Load Weights
    observer.load_state_dict(torch.load(os.path.join(config.BASE_DIR, "observer.pt"), map_location=device, weights_only=True))
    quantizer.load_state_dict(torch.load(os.path.join(config.BASE_DIR, "quantizer.pt"), map_location=device, weights_only=True))
    
    # Freeze Weights
    observer.eval()
    quantizer.eval()
    for p in observer.parameters(): p.requires_grad = False
    for p in quantizer.parameters(): p.requires_grad = False
    
    generator = LatentGenerator(
        d_model=config.D_MODEL, latent_len=config.LATENT_LEN, 
        nhead=config.N_HEADS, num_layers=8, 
        codebook_size=config.CODEBOOK_SIZE, num_quantizers=config.NUM_QUANTIZERS
    ).to(device)
    
    optimizer = optim.Adam(generator.parameters(), lr=3e-4)
    
    for epoch in range(1, config.PHASE2_EPOCHS + 1):
        generator.train()
        total_loss = 0.0
        total_acc_s1, total_acc_s2 = 0.0, 0.0
        start_time = time.time()
        
        for batch_idx, (curr_phrase, next_phrase) in enumerate(dataloader):
            curr_phrase, next_phrase = curr_phrase.to(device), next_phrase.to(device)
            
            # Extract discrete latents using frozen Phase 1
            with torch.no_grad():
                z_curr = observer(curr_phrase)
                _, indices_curr, _ = quantizer(z_curr) # (B, LATENT_LEN, NUM_QUANTIZERS)
                
                z_next = observer(next_phrase)
                _, indices_next, _ = quantizer(z_next) # (B, LATENT_LEN, NUM_QUANTIZERS)
                
            # Concatenate curr and next to form a continuous sequence of latents
            # full_seq shape: (B, LATENT_LEN * 2, NUM_QUANTIZERS)
            full_seq = torch.cat([indices_curr, indices_next], dim=1)
            
            # Causal Language Modeling: predict the next token given all previous
            inputs = full_seq[:, :-1, :]  # Shape: (B, seq_len-1, 2)
            targets = full_seq[:, 1:, :]  # Shape: (B, seq_len-1, 2)
            
            optimizer.zero_grad()
            logits = generator(inputs)    # Shape: (B, seq_len-1, 2048, 2)
            
            # --- CRITICAL BUG FIX ---
            B, L, C, Q = logits.shape
            
            # Permute Q to align with B and L BEFORE flattening C
            logits_permuted = logits.permute(0, 1, 3, 2).contiguous() # (B, L, 2, 2048)
            logits_flat = logits_permuted.reshape(-1, C)              # (B * L * 2, 2048)
            
            targets_flat = targets.reshape(-1)                        # (B * L * 2)
            
            loss = F.cross_entropy(logits_flat, targets_flat)
            loss.backward()
            optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=2) # (B, seq_len-1, 2)
            
            acc_s1 = (preds[..., 0] == targets[..., 0]).float().mean().item()
            acc_s2 = (preds[..., 1] == targets[..., 1]).float().mean().item()
            
            total_acc_s1 += acc_s1
            total_acc_s2 += acc_s2
            
        avg_loss = total_loss / len(dataloader)
        avg_acc_s1 = (total_acc_s1 / len(dataloader)) * 100
        avg_acc_s2 = (total_acc_s2 / len(dataloader)) * 100
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch:02d}/{config.PHASE2_EPOCHS} | "
              f"Time: {elapsed:.2f}s | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc S1: {avg_acc_s1:.2f}% | "
              f"Acc S2: {avg_acc_s2:.2f}%")

    print("\n✅ Phase 2 Complete. Saving generator...")
    os.makedirs(config.BASE_DIR, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(config.BASE_DIR, "generator.pt"))
    print("Phase 2 Generator fully trained and saved!")

if __name__ == "__main__":
    main()