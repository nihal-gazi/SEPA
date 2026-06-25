"""
models.py — GSCv3 Neural Architecture

Contains the Phase 1 Autoencoder (Observer, RVQ, Reconstructor)
and the Phase 2 Autoregressive Latent Generator.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Phase 1: VQ-Autoencoder Components
# ============================================================

class Observer(nn.Module):
    """Encodes the 1D Interpolated Text into Continuous Latents."""
    def __init__(self, vocab_size, d_model, seq_len, latent_len, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        # Local Chunking: Compress seq_len (16) to latent_len (8)
        stride = seq_len // latent_len
        self.compress = nn.Conv1d(d_model, d_model, kernel_size=stride, stride=stride)

    def forward(self, x):
        # x shape: (B, seq_len)
        x = self.embedding(x) + self.pos_emb
        x = self.transformer(x)               # (B, seq_len, d_model)
        
        x = x.transpose(1, 2)                 # (B, d_model, seq_len)
        x = self.compress(x)                  # (B, d_model, latent_len)
        x = x.transpose(1, 2)                 # (B, latent_len, d_model)
        return x


class EMAVectorQuantizer(nn.Module):
    """Single stage VQ with Exponential Moving Average and Dead Code Revival."""
    def __init__(self, d_model, codebook_size, codebook_dim, decay=0.99):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        
        self.proj_in = nn.Linear(d_model, codebook_dim)
        self.proj_out = nn.Linear(codebook_dim, d_model)
        
        # EMA states
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', torch.randn(codebook_size, codebook_dim))
        self.codebook = nn.Parameter(torch.randn(codebook_size, codebook_dim), requires_grad=False)

    def forward(self, x):
        z = self.proj_in(x) # (B, L, codebook_dim)
        flat_z = z.view(-1, self.codebook_dim)
        
        # L2 Distances
        distances = (torch.sum(flat_z**2, dim=1, keepdim=True) 
                    + torch.sum(self.codebook**2, dim=1)
                    - 2 * torch.matmul(flat_z, self.codebook.t()))
        
        indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(indices, self.codebook).view(z.shape)
        
        if self.training:
            # EMA Update
            indices_onehot = F.one_hot(indices, self.codebook_size).float()
            self.cluster_size.data.mul_(self.decay).add_(indices_onehot.sum(0), alpha=1 - self.decay)
            
            embed_sum = torch.matmul(indices_onehot.t(), flat_z.detach())
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            
            # Dead code revival
            usage = (self.cluster_size >= 1.0).float().unsqueeze(-1)
            random_z = flat_z[torch.randint(0, flat_z.size(0), (self.codebook_size,))]
            
            self.embed_avg.data = usage * self.embed_avg.data + (1 - usage) * random_z
            self.cluster_size.data = usage.squeeze() * self.cluster_size.data + (1 - usage.squeeze()) * 1.0
            
            self.codebook.data = self.embed_avg / self.cluster_size.unsqueeze(-1)
        
        # Straight-Through Estimator (STE)
        quantized_ste = z + (quantized - z).detach()
        commitment_loss = F.mse_loss(z, quantized.detach())
        
        out = self.proj_out(quantized_ste)
        return out, indices.view(x.shape[0], x.shape[1]), commitment_loss


class ResidualVectorQuantizer(nn.Module):
    """Multi-stage RVQ Pipeline."""
    def __init__(self, d_model, num_quantizers, codebook_size, codebook_dim):
        super().__init__()
        self.quantizers = nn.ModuleList([
            EMAVectorQuantizer(d_model, codebook_size, codebook_dim) 
            for _ in range(num_quantizers)
        ])
        
    def forward(self, x):
        quantized_out = 0
        indices_list = []
        commitment_loss = 0
        
        residual = x
        for q in self.quantizers:
            q_out, indices, loss = q(residual)
            quantized_out = quantized_out + q_out
            residual = residual - q_out
            indices_list.append(indices)
            commitment_loss += loss
            
        indices_stacked = torch.stack(indices_list, dim=-1) # (B, L, num_quantizers)
        return quantized_out, indices_stacked, commitment_loss
        
    def get_quantized_projection(self, indices):
        """Used during Inference to map discrete indices back to continuous vectors."""
        quantized_out = 0
        for i, q in enumerate(self.quantizers):
            idx = indices[..., i]
            quantized = F.embedding(idx, q.codebook)
            quantized_out += q.proj_out(quantized)
        return quantized_out


class Reconstructor(nn.Module):
    """Decodes Continuous Latents back into Interpolated Text."""
    def __init__(self, vocab_size, embedding_weight, d_model, seq_len, latent_len, nhead, num_layers):
        super().__init__()
        stride = seq_len // latent_len
        # Expansion: latent_len (8) -> seq_len (16)
        self.expand = nn.ConvTranspose1d(d_model, d_model, kernel_size=stride, stride=stride)
        
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        self.out_proj = nn.Linear(d_model, vocab_size)
        
        # Selective Weight Tying: Share vocab embeddings with Observer if provided
        if embedding_weight is not None:
            self.out_proj.weight = embedding_weight

    def forward(self, x):
        # x: (B, latent_len, d_model)
        x = x.transpose(1, 2)                 # (B, d_model, latent_len)
        x = self.expand(x)                    # (B, d_model, seq_len)
        x = x.transpose(1, 2)                 # (B, seq_len, d_model)
        
        x = x + self.pos_emb
        x = self.transformer(x)
        logits = self.out_proj(x)
        return logits


# ============================================================
# Phase 2: Autoregressive Latent Generator
# ============================================================

class LatentGenerator(nn.Module):
    """Causal Transformer predicting the next block of Discrete Latents."""
    def __init__(self, d_model, latent_len, nhead, num_layers, codebook_size, num_quantizers):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        
        # Independent embeddings for each RVQ stage
        self.embeddings = nn.ModuleList([
            nn.Embedding(codebook_size, d_model) for _ in range(num_quantizers)
        ])
        
        # Positional Embedding (supports expanded sliding window up to 2x latent_len)
        self.pos_emb = nn.Parameter(torch.zeros(1, latent_len * 2, d_model))
        
        # Causal Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        # Output projections
        self.out_projs = nn.ModuleList([
            nn.Linear(d_model, codebook_size) for _ in range(num_quantizers)
        ])
        
        # Massive optimization: Tie Generator input/output codebook weights
        for i in range(num_quantizers):
            self.out_projs[i].weight = self.embeddings[i].weight

    def forward(self, indices):
        # indices shape: (B, sequence_length, num_quantizers)
        seq_len = indices.size(1)
        
        # Aggregate embeddings across RVQ stages
        x = 0
        for i in range(self.num_quantizers):
            x = x + self.embeddings[i](indices[..., i])
            
        x = x + self.pos_emb[:, :seq_len, :]
        
        # Create additive causal mask (-inf for future tokens)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=x.device), diagonal=1)
        
        x = self.transformer(x, mask=mask) # (B, seq_len, d_model)
        
        # Predict logits for each stage
        logits = []
        for i in range(self.num_quantizers):
            logits.append(self.out_projs[i](x))
            
        return torch.stack(logits, dim=-1) # (B, seq_len, codebook_size, num_quantizers)