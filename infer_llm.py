"""
infer_llm.py — Baseline LLM Inference

Autoregressive token-by-token generation using the standard baseline model.
Matches the prompts and top-p sampling used in the GSC comparison.
"""

import os
import sys
import torch
import torch.nn.functional as F
import re

from dataset_preprocess import get_or_train_tokenizer
from LLM import config
from LLM.model import BaselineLLM

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

@torch.no_grad()
def generate(prompt, model, tokenizer, device, max_new_tokens=40, temp=0.7, top_p=0.9):
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    
    for _ in range(max_new_tokens):
        # Truncate context to max_seq_len if it gets too long
        if input_ids.size(1) > config.MAX_SEQ_LEN:
            context = input_ids[:, -config.MAX_SEQ_LEN:]
        else:
            context = input_ids
            
        logits = model(context)
        next_token_logits = logits[0, -1, :] # Grab the last predicted token logits
        
        next_token_id = sample_top_p(next_token_logits, temp=temp, top_p=top_p)
        
        # Stop early if we hit an end of text token
        if next_token_id == tokenizer.token_to_id("<|endoftext|>"):
            break
            
        # Append and continue
        next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    output_text = tokenizer.decode(input_ids[0].tolist())
    return clean_text_formatting(output_text)

def main():
    device = config.DEVICE
    print("Loading Baseline LLM...")
    
    tokenizer = get_or_train_tokenizer(raw_text_path=config.DATASET_PATH, vocab_size=config.VOCAB_SIZE)
    
    model = BaselineLLM(
        vocab_size=config.VOCAB_SIZE, 
        d_model=config.D_MODEL, 
        n_heads=config.N_HEADS, 
        num_layers=config.NUM_LAYERS, 
        dim_feedforward=config.DIM_FEEDFORWARD, 
        max_seq_len=config.MAX_SEQ_LEN
    ).to(device)
    
    model_path = os.path.join(config.BASE_DIR, "baseline_llm.pt")
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run train_llm.py first.")
        sys.exit(1)
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    prompts = [
        "Once upon a time, there was a little girl named Lily.",
        "A small boy found a shiny red ball in the garden.",
        "The cat wanted to climb the tall green tree.",
        "Mom said it was time to go to the park.",
        "There was a big dog who loved to play with his friends.",
    ]

    for temp in [0.4, 0.7]:
        print(f"\n{'='*60}")
        print(f"  BASELINE LLM | Temperature = {temp} (Top-P = 0.90)")
        print(f"{'='*60}")

        for i, prompt in enumerate(prompts):
            output = generate(prompt, model, tokenizer, device, max_new_tokens=60, temp=temp, top_p=0.90)
            print(f"\n[{i+1}] {output}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()