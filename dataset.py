"""
dataset.py — GSCv3 Preprocessing Pipeline (Clean & From Scratch)

This script reads the raw TinyStories dataset, splits it by <|endoftext|>, 
and packages it into a clean JSON format.

Instead of risky dependency-tree parsing that drops or scrambles words, 
this uses safe, contiguous sentence-level splitting to guarantee 
100% of the narrative flow is preserved exactly as written.
"""

import os
import json
import re

def process_into_chunks(text):
    """
    Safely splits the text into contiguous chunks (sentences) 
    without dropping or reordering any words.
    """
    # Clean up whitespace and newlines
    text = text.replace('\n', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    
    # Simple, safe sentence splitting using regex on punctuation
    # This splits on ., !, or ? followed by a space and an uppercase letter
    raw_chunks = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    clean_chunks = [chunk.strip() for chunk in raw_chunks if chunk.strip()]
    return clean_chunks

def main():
    input_file = "dataset_10.txt"
    output_file = "processed_dataset.json"
    
    if not os.path.exists(input_file):
        print(f"Error: Could not find '{input_file}' in the current directory.")
        return

    print(f"Reading '{input_file}'...")
    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Strictly split by the dataset delimiter
    raw_stories = raw_text.split("<|endoftext|>")
    
    processed_stories = []
    
    for i, story in enumerate(raw_stories):
        clean_story = story.strip()
        if not clean_story:
            continue
            
        # Extract safe, contiguous phrases (sentences)
        phrases = process_into_chunks(clean_story)
        
        if phrases:
            processed_stories.append({
                "id": i,
                "phrases": phrases
            })

    print(f"Found and processed {len(processed_stories)} stories.")
    print(f"Saving to '{output_file}'...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_stories, f, indent=2, ensure_ascii=False)
        
    print("Done! Dataset successfully converted to clean JSON.")

if __name__ == "__main__":
    main()