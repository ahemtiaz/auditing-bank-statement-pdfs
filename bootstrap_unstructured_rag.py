import sys
import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Setup paths
workspace_dir = Path(__file__).parent.resolve()
data_dir = workspace_dir / "data" / "bank-statements"
output_dir = workspace_dir / "output" / "parsing" / "unstructured_rag"

# Import from unified rag_helper package
try:
    from rag_helper import PdfPlumberParser, neural_chunker, download_retriever_models
except ImportError as e:
    print(f"[ERROR] Failed to import from rag_helper: {e}")
    sys.exit(1)

def run_download_models():
    print("\n--- Phase 3: Verifying/Downloading Retriever Models ---")
    try:
        download_retriever_models()
    except Exception as e:
        print(f"[FATAL ERROR] Failed to download retriever models: {e}")
        sys.exit(1)

def main():
    print("=== Starting RAG-1 (Unstructured) Bootstrap Process ===")
    
    csv_path = output_dir / "chunk_corpus.csv"
    if csv_path.exists():
        print(f"RAG-1 chunk corpus already exists at '{csv_path}'. Skipping parsing and chunking phases.")
        run_download_models()
        print("\n=== Bootstrap Setup Complete! ===")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not data_dir.exists():
        print(f"[ERROR] Data directory '{data_dir}' does not exist.")
        sys.exit(1)
        
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[ERROR] No PDF files found in '{data_dir}'.")
        sys.exit(1)
        
    print(f"Found {len(pdf_files)} PDF bank statement(s) to process.")
    
    # 2. Parse PDFs using PdfPlumberParser
    print("\n--- Phase 1: Parsing PDFs using pdfplumber ---")
    parser = PdfPlumberParser(results_dir=output_dir)
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"[{i}/{len(pdf_files)}] Parsing '{pdf_path.name}'...")
        parser.partition_pdf(pdf_path)
        
    print("\nParsing completed!")
    print(parser.get_timing_summary())
    
    # 3. Chunking Parsed Text
    print("\n--- Phase 2: Chunking Text with 0 Overlap ---")
    parsed_dir = output_dir / "pdfplumber"
    
    corpus_chunks = []
    
    # Find all parsed JSON files
    json_files = list(parsed_dir.glob("*_pdfplumber.json"))
    if not json_files:
        print(f"[ERROR] No parsed JSON files found in '{parsed_dir}'. Make sure parsing succeeded.")
        sys.exit(1)
        
    chunk_size = 512
    chunk_overlap = 0  # 0 overlap to avoid duplicate transactions
    
    for i, json_path in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] Chunking parsed file '{json_path.name}'...")
        
        # Load parsed pages
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            pages = json.load(f)
            
        file_stem = json_path.name.replace("_pdfplumber.json", "")
        
        # Process each page
        for page_data in pages:
            page_num = page_data.get('page_number', 1)
            page_text = page_data.get('text', '').strip()
            
            if not page_text:
                continue
                
            # Chunk the page text
            try:
                chunks = neural_chunker(
                    text=page_text,
                    tokenizer='gpt2',
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    min_characters_per_chunk=24
                )
                
                # Add each chunk to the corpus
                for chunk_idx, chunk_text in enumerate(chunks, 1):
                    chunk_id = f"{file_stem}_page_{page_num}_chunk_{chunk_idx}"
                    image_filename = f"{file_stem}_page_{page_num}.png"
                    
                    corpus_chunks.append({
                        "chunk_id": chunk_id,
                        "text_description": chunk_text,
                        "image_filename": image_filename,
                        "query": ""  # compatible with DocumentProvider schema
                    })
            except Exception as e:
                print(f"[ERROR] Failed to chunk page {page_num} of {file_stem}: {e}")
                
    print(f"\nChunking completed! Generated {len(corpus_chunks)} total chunks.")
    
    # Save to CSV
    df = pd.DataFrame(corpus_chunks)
    df.to_csv(csv_path, index=False)
    print(f"Saved chunk corpus to: {csv_path}")
    
    # Run model downloading
    run_download_models()
    
    print("\n=== Bootstrap Setup Complete! ===")

if __name__ == "__main__":
    main()
