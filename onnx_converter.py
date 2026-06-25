"""
onnx_converter.py — Exports the PyTorch GSCv3 models to ONNX Web format.

We wrap the models into 3 logical blocks (Encoder, Generator, Decoder)
to make the Javascript execution loop as clean and fast as possible.
"""

import os
import sys
import shutil
import torch
import torch.nn as nn

import config
from models import Observer, ResidualVectorQuantizer, Reconstructor, LatentGenerator

# 1. Wrapper Modules to streamline ONNX inputs/outputs
class GSCEncoder(nn.Module):
    def __init__(self, observer, quantizer):
        super().__init__()
        self.observer = observer
        self.quantizer = quantizer
    def forward(self, x):
        z = self.observer(x)
        _, indices, _ = self.quantizer(z)
        return indices # Shape: (1, LATENT_LEN, 2)

class GSCDecoder(nn.Module):
    def __init__(self, quantizer, reconstructor):
        super().__init__()
        self.quantizer = quantizer
        self.reconstructor = reconstructor
    def forward(self, indices):
        q = self.quantizer.get_quantized_projection(indices)
        logits = self.reconstructor(q)
        return logits # Shape: (1, SEQ_LEN, VOCAB_SIZE)

def main():
    device = "cpu" # Exporting is safer and easier on CPU
    print("Loading PyTorch Models...")
    
    # Initialize Models
    observer = Observer(vocab_size=config.VOCAB_SIZE, d_model=config.D_MODEL, seq_len=config.SEQ_LEN, latent_len=config.LATENT_LEN, nhead=config.N_HEADS, num_layers=6).to(device)
    quantizer = ResidualVectorQuantizer(d_model=config.D_MODEL, num_quantizers=config.NUM_QUANTIZERS, codebook_size=config.CODEBOOK_SIZE, codebook_dim=config.CODEBOOK_DIM).to(device)
    reconstructor = Reconstructor(vocab_size=config.VOCAB_SIZE, embedding_weight=observer.embedding.weight, d_model=config.D_MODEL, seq_len=config.SEQ_LEN, latent_len=config.LATENT_LEN, nhead=config.N_HEADS, num_layers=4).to(device)
    generator = LatentGenerator(d_model=config.D_MODEL, latent_len=config.LATENT_LEN, nhead=config.N_HEADS, num_layers=8, codebook_size=config.CODEBOOK_SIZE, num_quantizers=config.NUM_QUANTIZERS).to(device)

    # Load Weights
    observer.load_state_dict(torch.load(os.path.join(config.BASE_DIR, "observer.pt"), map_location=device, weights_only=True))
    quantizer.load_state_dict(torch.load(os.path.join(config.BASE_DIR, "quantizer.pt"), map_location=device, weights_only=True))
    reconstructor.load_state_dict(torch.load(os.path.join(config.BASE_DIR, "reconstructor.pt"), map_location=device, weights_only=True))
    generator.load_state_dict(torch.load(os.path.join(config.BASE_DIR, "generator.pt"), map_location=device, weights_only=True))

    observer.eval()
    quantizer.eval()
    reconstructor.eval()
    generator.eval()

    # Create directories
    os.makedirs("web_server/models", exist_ok=True)

    print("Exporting Encoder to ONNX...")
    encoder = GSCEncoder(observer, quantizer).eval()
    dummy_text = torch.zeros(1, config.SEQ_LEN, dtype=torch.long)
    torch.onnx.export(encoder, dummy_text, "web_server/models/encoder.onnx", 
                      input_names=['input_ids'], output_names=['latent_indices'], opset_version=14)

    print("Exporting Decoder to ONNX...")
    decoder = GSCDecoder(quantizer, reconstructor).eval()
    dummy_latents = torch.zeros(1, config.LATENT_LEN, 2, dtype=torch.long)
    torch.onnx.export(decoder, dummy_latents, "web_server/models/decoder.onnx", 
                      input_names=['generated_latents'], output_names=['text_logits'], opset_version=14)

    print("Exporting Generator to ONNX (Fixed size to prevent WASM crash)...")
    # Set the dummy context to the maximum sliding window size (32)
    dummy_context = torch.zeros(1, config.LATENT_LEN * 2, 2, dtype=torch.long)
    torch.onnx.export(generator, dummy_context, "web_server/models/generator.onnx", 
                      input_names=['context'], output_names=['logits'],
                      # We REMOVED dynamic_axes to lock the shape to exactly 32
                      opset_version=14)

    # Copy the valid dictionary over for JS to use
    print("Copying valid dictionary to web server...")
    if os.path.exists("valid_pairs.json"):
        shutil.copy("valid_pairs.json", "web_server/valid_pairs.json")

    print("\n✅ All models successfully exported to ONNX format in 'web_server/models/'")

if __name__ == "__main__":
    main()