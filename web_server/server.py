"""
server.py — Flask Backend API & Static Host

Serves the Web UI and models locally. Exposes lightweight tokenization APIs
so the browser doesn't have to reimplement Python's BPE parsing logic.
"""

import os
import sys
import itertools
import re
from flask import Flask, request, jsonify, send_from_directory

# Allows server to import scripts from the parent GSCv3 directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataset_preprocess import get_or_train_tokenizer, interpolate_phrase

app = Flask(__name__, static_folder='.')

# Load Tokenizer once
tokenizer = get_or_train_tokenizer(raw_text_path="../dataset_10.txt", vocab_size=config.VOCAB_SIZE)
eop_id = tokenizer.token_to_id("<|endofphrase|>")

def process_into_chunks(text):
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    raw_chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [chunk.strip() for chunk in raw_chunks if chunk.strip()]

def clean_text_formatting(text):
    text = text.replace('Ġ', ' ') 
    text = re.sub(r'\s+([?.!,:;])', r'\1', text) 
    text = re.sub(r'(["\'])\s+', r'\1', text)    
    text = re.sub(r'\s+(["\'])', r'\1', text)
    text = text.replace(" 's", "'s").replace(" n't", "n't").replace(" 're", "'re").replace(" 'm", "'m")
    return re.sub(r'\s+', ' ', text).strip()

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/api/tokenize', methods=['POST'])
def api_tokenize():
    """Takes a prompt, extracts the last sentence, and pads to SEQ_LEN."""
    data = request.json
    prompt = data.get("prompt", "")
    chunks = process_into_chunks(prompt)
    last_chunk = chunks[-1] if chunks else prompt
    
    tokens = tokenizer.encode(last_chunk).ids + [eop_id]
    scaled = interpolate_phrase(tokens, config.SEQ_LEN)
    
    return jsonify({"tokens": scaled})

@app.route('/api/detokenize', methods=['POST'])
def api_detokenize():
    """Takes text logits, squashes 1D interpolation, and decodes to raw text."""
    data = request.json
    decoded_ids = data.get("decoded_ids", [])
    
    # Squash duplicates
    squashed = [tok_id for tok_id, _ in itertools.groupby(decoded_ids)]
    
    # Boundary Cut
    if eop_id in squashed:
        squashed = squashed[:squashed.index(eop_id)]
        
    if not squashed:
        return jsonify({"text": ""})
        
    phrase_text = tokenizer.decode(squashed).replace('Ġ', ' ').strip()
    clean_phrase = clean_text_formatting(phrase_text)
    
    return jsonify({"text": clean_phrase})

if __name__ == '__main__':
    # Listen on all interfaces (0.0.0.0) so a phone on Wi-Fi can connect
    print("Starting Server on port 5000...")
    print("Find your PC's local IP address (e.g., 192.168.1.X) and visit http://<YOUR_IP>:5000 on your phone!")
    app.run(host='0.0.0.0', port=5000)