"""
infer.py — SEPA Inference Engine

Autoregressive Latent Generation with Nucleus (Top-P) Sampling, 
Valid Codebook Masking, 1D Interpolation Decoding, and Auto-Formatting.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import itertools
import re

# Ensure Python can find the SEPA module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from dataset_preprocess import get_or_train_tokenizer, interpolate_phrase
from models import Observer, ResidualVectorQuantizer, Reconstructor, LatentGenerator

def clean_text_formatting(text):
    """Fixes BPE spacing artifacts around punctuation."""
    text = text.replace('Ġ', ' ') 
    text = re.sub(r'\s+([?.!,:;])', r'\1', text) 
    text = re.sub(r'(["\'])\s+', r'\1', text)    
    text = re.sub(r'\s+(["\'])', r'\1', text)
    text = text.replace(" 's", "'s").replace(" n't", "n't").replace(" 're", "'re").replace(" 'm", "'m")
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def sample_top_p(logits, temp=0.7, top_p=0.9):
    """Nucleus sampling: dynamically truncates the noise tail."""
    if temp == 0.0:
        return torch.argmax(logits).item()
    
    logits = logits / temp
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

def load_models(device):
    print("Loading SEPA Tokenizer...")
    tokenizer = get_or_train_tokenizer(raw_text_path="dataset_10.txt", vocab_size=config.VOCAB_SIZE)

    print("Initializing SEPA Architecture...")
    observer = Observer(
        vocab_size=config.VOCAB_SIZE, d_model=config.D_MODEL,
        seq_len=config.SEQ_LEN, latent_len=config.LATENT_LEN,
        nhead=config.N_HEADS, num_layers=6,
    ).to(device)

    quantizer = ResidualVectorQuantizer(
        d_model=config.D_MODEL, num_quantizers=config.NUM_QUANTIZERS,
        codebook_size=config.CODEBOOK_SIZE, codebook_dim=config.CODEBOOK_DIM,
    ).to(device)

    reconstructor = Reconstructor(
        vocab_size=config.VOCAB_SIZE,
        embedding_weight=observer.embedding.weight,
        d_model=config.D_MODEL, seq_len=config.SEQ_LEN, latent_len=config.LATENT_LEN,
        nhead=config.N_HEADS, num_layers=4,
    ).to(device)

    # Use exact decoupled generator dimensions if specified, else fallback to base d_model
    gen_d_model = getattr(config, 'GEN_D_MODEL', config.D_MODEL)
    gen_n_heads = getattr(config, 'GEN_N_HEADS', config.N_HEADS)
    gen_layers = getattr(config, 'GEN_LAYERS', 8)

    generator = LatentGenerator(
        d_model=gen_d_model, latent_len=config.LATENT_LEN,
        nhead=gen_n_heads, num_layers=gen_layers,
        codebook_size=config.CODEBOOK_SIZE,
        num_quantizers=config.NUM_QUANTIZERS,
    ).to(device)

    print("Loading Checkpoint Weights...")
    observer.load_state_dict(torch.load(os.path.join(config.BASE_DIR, "observer.pt"), map_location=device, weights_only=True))
    quantizer.load_state_dict(torch.load(os.path.join(config.BASE_DIR, "quantizer.pt"), map_location=device, weights_only=True))
    reconstructor.load_state_dict(torch.load(os.path.join(config.BASE_DIR, "reconstructor.pt"), map_location=device, weights_only=True))
    generator.load_state_dict(torch.load(os.path.join(config.BASE_DIR, "generator.pt"), map_location=device, weights_only=True))

    for m in (observer, quantizer, reconstructor, generator):
        m.eval()

    print("Loading Valid Codebook Dictionary...")
    with open(os.path.join(config.BASE_DIR, "valid_pairs.json"), "r") as f:
        raw = json.load(f)
    valid_pairs = {int(k): set(v) for k, v in raw.items()}

    return observer, quantizer, reconstructor, generator, tokenizer, valid_pairs

def process_into_chunks(text):
    """
    CRITICAL FIX: 1:1 mathematical match with the Web API logic.
    Splits by full sentences so the Autoencoder stays strictly in-distribution.
    """
    clean_story = text.replace('\n', ' ').strip()
    clean_story = re.sub(r'\s+', ' ', clean_story)
    raw_chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean_story)
    phrases = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
    return phrases

@torch.no_grad()
def generate_sepa(prompt, observer, quantizer, reconstructor, generator,
                  tokenizer, valid_pairs, device, num_phrases=10, temp=0.7, top_p=0.9):

    eop_id = tokenizer.token_to_id("<|endofphrase|>")
    generated_text = prompt

    for _ in range(num_phrases):
        
        # Extract the last sentence of the entire accumulated story to encode
        chunks = process_into_chunks(generated_text)
        last_phrase = chunks[-1] if chunks else generated_text
        
        tokens = tokenizer.encode(last_phrase).ids + [eop_id]
        scaled = interpolate_phrase(tokens, config.SEQ_LEN)
        curr_block = torch.tensor([scaled], dtype=torch.long, device=device)

        # 2. Compress current block into latents
        z = observer(curr_block)
        _, indices_curr, _ = quantizer(z)             # Shape: (1, LATENT_LEN, 2)

        context = indices_curr.clone()
        gen_s1, gen_s2 = [], []
        
        # 3. Autoregressive Latent Generation
        for _ in range(config.LATENT_LEN):
            logits = generator(context)               
            step_logits = logits[0, -1, :, :]         # Next token logic: (2048, 2)

            # Stage 1 — Free Nucleus Sampling
            s1_idx = sample_top_p(step_logits[:, 0], temp=temp, top_p=top_p)

            # Stage 2 — MASKED Nucleus Sampling
            s2_logits = step_logits[:, 1].clone()
            allowed = valid_pairs.get(s1_idx)
            if allowed is not None and len(allowed) < config.CODEBOOK_SIZE:
                mask = torch.ones(config.CODEBOOK_SIZE, dtype=torch.bool, device=device)
                mask[list(allowed)] = False
                s2_logits[mask] = float("-inf") # Block out-of-distribution combos

            s2_idx = sample_top_p(s2_logits, temp=temp, top_p=top_p)

            gen_s1.append(s1_idx)
            gen_s2.append(s2_idx)
            
            # Slide Context Window
            next_latent = torch.tensor([[[s1_idx, s2_idx]]], dtype=torch.long, device=device)
            context = torch.cat([context, next_latent], dim=1)
            if context.shape[1] > config.LATENT_LEN * 2:
                context = context[:, -(config.LATENT_LEN * 2):, :]

        # 4. Full-Block 1D Decode
        next_indices = torch.tensor([[gen_s1, gen_s2]], dtype=torch.long, device=device).permute(0, 2, 1)
        q_next = quantizer.get_quantized_projection(next_indices)
        token_logits = reconstructor(q_next)            
        decoded_ids = torch.argmax(token_logits, dim=-1)[0].tolist()

        # 5. De-Interpolation Squash
        squashed = [tok_id for tok_id, _ in itertools.groupby(decoded_ids)]

        # 6. Semantic Cut
        if eop_id in squashed:
            squashed = squashed[:squashed.index(eop_id)]

        if not squashed:
            continue

        # Strip the BPE characters cleanly
        phrase_text = tokenizer.decode(squashed).replace('Ġ', ' ').replace('  ', ' ').strip()
        
        # --- THE FIX: Align Python Tokenization format with JS by cleaning mid-loop ---
        phrase_text = clean_text_formatting(phrase_text) 
        
        if not phrase_text:
            continue

        # 7. Append directly to story (Exactly like Javascript UI)
        if (generated_text
            and not generated_text.endswith(" ")
            and not phrase_text.startswith(" ")
            and not phrase_text.startswith(".")
            and not phrase_text.startswith(",")
            and not phrase_text.startswith("!")
            and not phrase_text.startswith("?")):
            generated_text += " "
        generated_text += phrase_text

    return clean_text_formatting(generated_text)

def main():
    device = config.DEVICE
    print(f"Using device: {device}")
    
    observer, quantizer, reconstructor, generator, tokenizer, valid_pairs = load_models(device)

    prompts = [
        "Once upon a time, there was a little girl named Lily.",
        "A small boy found a shiny red ball in the garden.",
        "The cat wanted to climb the tall green tree.",
        "Mom said it was time to go to the park.",
        "There was a big dog who loved to play with his friends.",
    ]

    # Added Temp = 0.0 to verify it perfectly matches the A06 Phone output!
    for temp in [0.0, 0.4, 0.7]:
        print(f"\n{'='*60}")
        print(f"  SEPA ENGINE | Temperature = {temp} (Top-P = 0.90)")
        print(f"{'='*60}")

        for i, prompt in enumerate(prompts):
            output = generate_sepa(
                prompt, observer, quantizer, reconstructor, generator,
                tokenizer, valid_pairs, device,
                num_phrases=10, temp=temp, top_p=0.90
            )
            print(f"\n[{i+1}] {output}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()