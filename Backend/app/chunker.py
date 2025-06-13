import os
import re

def chunk_text(extracted_text, doc_name="Unknown Document", page_num=1):
    # Split into paragraphs first
    paragraphs = [p.strip() for p in extracted_text.split("\n\n") if p.strip()]
    chunks = []
    
    # Process each paragraph
    for para_idx, paragraph in enumerate(paragraphs, start=1):
        # If paragraph is short, keep it as one chunk
        if len(paragraph.split()) < 50:
            chunks.append({
                "chunk_text": paragraph,
                "document": doc_name,
                "page": page_num,
                "paragraph": para_idx,
                "sentence": 1
            })
            continue
            
        # For longer paragraphs, split into sentences but keep more context
        sentences = re.split(r'(?<=[.!?]) +', paragraph)
        current_chunk = []
        current_length = 0
        
        for sent_idx, sentence in enumerate(sentences, start=1):
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += len(sentence.split())
            
            # If chunk is long enough or this is the last sentence, save it
            if current_length >= 100 or sent_idx == len(sentences):
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "chunk_text": chunk_text,
                    "document": doc_name,
                    "page": page_num,
                    "paragraph": para_idx,
                    "sentence": sent_idx
                })
                current_chunk = []
                current_length = 0
    
    return chunks
