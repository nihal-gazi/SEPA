"""
model.py — Baseline Causal LLM Architecture

A standard Decoder-only Causal Transformer (like GPT-2/LLaMA) scaled down
to match the GSC footprint. Uses Weight Tying for efficiency.
"""

import torch
import torch.nn as nn

class BaselineLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, dim_feedforward, max_seq_len):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Standard Transformer Encoder configured as a Causal Decoder
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=dim_feedforward, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Weight Tying: Share input embedding and output projection weights
        self.lm_head.weight = self.embed.weight

    def forward(self, x):
        # x shape: (Batch, Sequence Length)
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        
        x = self.embed(x) + self.pos_emb(positions)
        
        # Create additive causal mask (-inf for future tokens)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        
        # Pass through causal transformer
        x = self.transformer(x, mask=mask, is_causal=True)
        
        # Project to vocabulary
        logits = self.lm_head(x) # (Batch, Sequence Length, Vocab Size)
        return logits