import tiktoken
import os
import json
from typing import List

def apply_overlap_to_chunks(chunks: List[str], overlap_size: int, tokenizer) -> List[str]:
    """
    Apply overlap to chunks that don't natively support it.
    
    Args:
        chunks: List of text chunks
        overlap_size: Number of tokens to overlap between chunks
        tokenizer: Tokenizer to use for counting tokens
    
    Returns:
        List of chunks with overlap applied
    """
    if overlap_size <= 0 or len(chunks) <= 1:
        return chunks
    
    overlapped_chunks = []
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            # First chunk remains unchanged
            overlapped_chunks.append(chunk)
        else:
            # Get tokens from previous chunk for overlap
            prev_chunk = chunks[i-1]
            prev_tokens = tokenizer.encode(prev_chunk)
            current_tokens = tokenizer.encode(chunk)
            
            # Take last 'overlap_size' tokens from previous chunk
            overlap_tokens = prev_tokens[-overlap_size:] if len(prev_tokens) > overlap_size else prev_tokens
            
            # Combine overlap with current chunk
            combined_tokens = overlap_tokens + current_tokens
            overlapped_text = tokenizer.decode(combined_tokens)
            
            overlapped_chunks.append(overlapped_text)
    
    return overlapped_chunks

def load_json_document(json_path):
    # Load the JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract the "text" field
    texts = [item["text"] for item in data]

    return texts[0]

def save_chunks_to_json(chunks: List[str], output_path: str) -> None:
    """
    Save a list of text chunks to a JSON file under key "chunks".

    Args:
        chunks: List of text chunk strings.
        output_path: Path where to write the JSON.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({'chunks': chunks}, f, ensure_ascii=False, indent=2)

def neural_chunker(text, tokenizer='gpt2', chunk_size=512, chunk_overlap=0, min_characters_per_chunk=24, output_path=None):
    chunks = []
    use_fallback = False
    
    try:
        from chonkie import NeuralChunker
        import torch
        tokenizer_obj = tiktoken.get_encoding(tokenizer)

        chunker = NeuralChunker(
            model="mirth/chonky_modernbert_base_1",  
            device_map="cuda" if torch.cuda.is_available() else "cpu",                         
            min_characters_per_chunk=10,             
            return_type="texts"                     
        )

        chunks = chunker.chunk(text)

        # Apply overlap if specified, because this chunker doesn't support overlap natively 
        if chunk_overlap > 0:
            chunks = apply_overlap_to_chunks(chunks, chunk_overlap, tokenizer_obj)
            
    except Exception as e:
        print(f"[INFO] Falling back to TokenChunker due to NeuralChunker/PyTorch load error: {e}")
        use_fallback = True
        
    if use_fallback:
        from chonkie import TokenChunker
        chunker = TokenChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks_objs = chunker.chunk(text)
        chunks = [c.text for c in chunks_objs]

    if output_path:
        save_chunks_to_json(chunks, output_path)

    print(len(chunks), "chunks created with chunk size", chunk_size, "and overlap", chunk_overlap, "and min_characters_per_chunk", min_characters_per_chunk)
  
    return chunks
