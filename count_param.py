"""
count_params.py — GSCv3 Parameter Counting Utility

Calculates the exact parameter footprint of the GSCv3 architecture, 
accounting for weight-tying optimizations.
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from models import Observer, ResidualVectorQuantizer, Reconstructor, LatentGenerator

def count_all_parameters(model):
    """Counts all parameters in a model, whether they require gradients or not."""
    return sum(p.numel() for p in model.parameters())

def main():
    print("Initializing Models based on config.py...\n")
    
    observer = Observer(
        vocab_size=config.VOCAB_SIZE, d_model=config.D_MODEL,
        seq_len=config.SEQ_LEN, latent_len=config.LATENT_LEN,
        nhead=config.N_HEADS, num_layers=6,
    )

    quantizer = ResidualVectorQuantizer(
        d_model=config.D_MODEL, num_quantizers=config.NUM_QUANTIZERS,
        codebook_size=config.CODEBOOK_SIZE, codebook_dim=config.CODEBOOK_DIM,
    )

    reconstructor = Reconstructor(
        vocab_size=config.VOCAB_SIZE,
        embedding_weight=observer.embedding.weight, # Tied Weight 1
        d_model=config.D_MODEL, seq_len=config.SEQ_LEN, latent_len=config.LATENT_LEN,
        nhead=config.N_HEADS, num_layers=4,
    )

    generator = LatentGenerator(
        d_model=config.D_MODEL, latent_len=config.LATENT_LEN,
        nhead=config.N_HEADS, num_layers=8,
        codebook_size=config.CODEBOOK_SIZE,
        num_quantizers=config.NUM_QUANTIZERS,
        # Generator handles Tied Weight 2 internally (embeddings = output proj)
    )

    obs_total = count_all_parameters(observer)
    quant_total = count_all_parameters(quantizer)
    recon_total = count_all_parameters(reconstructor)
    gen_total = count_all_parameters(generator)

    print(f"{'Model Component':<25} | {'Parameters':<15}")
    print("-" * 45)
    print(f"{'Observer':<25} | {obs_total:,}")
    print(f"{'Quantizer (RVQ)':<25} | {quant_total:,}")
    print(f"{'Reconstructor':<25} | {recon_total:,}")
    print(f"{'Latent Generator (AR)':<25} | {gen_total:,}")
    print("-" * 45)
    
    total_params_raw = obs_total + quant_total + recon_total + gen_total
    print(f"{'RAW TOTAL (Double-Counted)':<25} | {total_params_raw:,}")
    
    # Deduplicate tied weights to get the true unique parameter footprint
    # We use the unique memory ID of each parameter tensor
    all_params = (
        list(observer.parameters()) + 
        list(quantizer.parameters()) + 
        list(reconstructor.parameters()) + 
        list(generator.parameters())
    )
    unique_params = sum(p.numel() for p in {id(p): p for p in all_params}.values())
    
    print("-" * 45)
    print(f"{'TRUE UNIQUE PARAMETERS':<25} | {unique_params:,}")
    print("-" * 45)
    print(f"\nConfiguration Details:")
    print(f"- d_model: {config.D_MODEL}")
    print(f"- vocab_size: {config.VOCAB_SIZE}")
    print(f"- codebook_size: {config.CODEBOOK_SIZE} (x{config.NUM_QUANTIZERS} quantizers)")

if __name__ == "__main__":
    main()