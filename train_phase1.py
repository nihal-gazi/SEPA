"""
train_phase1.py — GSCv3 Phase 1 Training

Trains Observer, RVQ, and Reconstructor to compress and decode interpolated phrases.
"""

import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import config
from dataset_preprocess import get_or_train_tokenizer, Phase1Dataset
from models import Observer, ResidualVectorQuantizer, Reconstructor

def main():
    device = config.DEVICE
    print(f"Using device: {device}")
    
    tokenizer = get_or_train_tokenizer(raw_text_path="dataset_10.txt", vocab_size=config.VOCAB_SIZE)
    
    print("\n" + "="*50)
    print("🚀 STARTING PHASE 1: VQ-AUTOENCODER TRAINING")
    print("="*50)
    
    dataset = Phase1Dataset(config.DATASET_PATH, tokenizer, seq_len=config.SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    # Initialize Phase 1 Models
    observer = Observer(
        vocab_size=config.VOCAB_SIZE, d_model=config.D_MODEL, 
        seq_len=config.SEQ_LEN, latent_len=config.LATENT_LEN, 
        nhead=config.N_HEADS, num_layers=6
    ).to(device)
    
    quantizer = ResidualVectorQuantizer(
        d_model=config.D_MODEL, num_quantizers=config.NUM_QUANTIZERS, 
        codebook_size=config.CODEBOOK_SIZE, codebook_dim=config.CODEBOOK_DIM
    ).to(device)
    
    reconstructor = Reconstructor(
        vocab_size=config.VOCAB_SIZE, embedding_weight=observer.embedding.weight, 
        d_model=config.D_MODEL, seq_len=config.SEQ_LEN, 
        latent_len=config.LATENT_LEN, nhead=config.N_HEADS, num_layers=4
    ).to(device)
    
    # Fix the duplicate parameters warning caused by Weight Tying
    all_params = list(observer.parameters()) + list(quantizer.parameters()) + list(reconstructor.parameters())
    unique_params = list({id(p): p for p in all_params}.values())
    
    optimizer = optim.Adam(unique_params, lr=3e-4)
    
    for epoch in range(1, config.PHASE1_EPOCHS + 1):
        observer.train()
        quantizer.train()
        reconstructor.train()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        total_acc = 0.0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(device) # (B, SEQ_LEN)
            optimizer.zero_grad()
            
            # Forward Pass
            z = observer(batch)
            q, indices, commitment_loss = quantizer(z)
            logits = reconstructor(q) # (B, SEQ_LEN, VOCAB_SIZE)
            
            # Loss Calculation (Using .reshape universally to prevent contiguous errors)
            recon_loss = F.cross_entropy(logits.reshape(-1, config.VOCAB_SIZE), batch.reshape(-1))
            loss = recon_loss + 0.25 * commitment_loss # 0.25 is standard VQ commitment weight
            
            loss.backward()
            optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == batch).float().mean().item()
            total_acc += acc
            
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_acc = (total_acc / len(dataloader)) * 100
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch:02d}/{config.PHASE1_EPOCHS} | "
              f"Time: {elapsed:.2f}s | "
              f"Total Loss: {avg_loss:.4f} | "
              f"Recon Loss: {avg_recon:.4f} | "
              f"Acc: {avg_acc:.2f}%")
              
    print("\n✅ Phase 1 Complete. Saving checkpoints...")
    os.makedirs(config.BASE_DIR, exist_ok=True)
    torch.save(observer.state_dict(), os.path.join(config.BASE_DIR, "observer.pt"))
    torch.save(quantizer.state_dict(), os.path.join(config.BASE_DIR, "quantizer.pt"))
    torch.save(reconstructor.state_dict(), os.path.join(config.BASE_DIR, "reconstructor.pt"))
    print("All Phase 1 models fully trained and saved!")

if __name__ == "__main__":
    main()