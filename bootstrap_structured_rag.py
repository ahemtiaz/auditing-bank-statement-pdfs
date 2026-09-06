import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Setup paths
workspace_dir = Path(__file__).parent.resolve()
parsing_dir = workspace_dir / "output" / "parsing" / "custom_parser"
output_dir = workspace_dir / "output" / "parsing" / "structured_rag"

# Import from unified rag_helper package
try:
    from rag_helper import get_combined_dataframe, chunk_dataframe_to_csv, download_retriever_models
except ImportError as e:
    print(f"[ERROR] Failed to import from rag_helper: {e}")
    sys.exit(1)

def main():
    print("=== Starting RAG-2/3 (Structured) Bootstrap Process ===")
    
    csv_path = output_dir / "chunk_corpus.csv"
    if csv_path.exists():
        print(f"RAG-2/3 chunk corpus already exists at '{csv_path}'. Skipping regeneration.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading parsed Excel files from: {parsing_dir}")
    
    # Combine Excel Data
    try:
        combined_df = get_combined_dataframe(parsing_dir)
        print(f"Successfully combined Excel data. Total raw rows: {len(combined_df)}")
    except Exception as e:
        print(f"[FATAL ERROR] Failed to combine Excel sheets: {e}")
        sys.exit(1)
        
    # Chunk DataFrame into CSV chunks
    print("\n--- Phase 2: Chunking combined DataFrame to CSV format ---")
    chunk_size = 512
    print(f"Using max_tokens: {chunk_size} per CSV chunk (no overlap, no split transactions)")
    
    try:
        corpus_chunks = chunk_dataframe_to_csv(combined_df, max_tokens=chunk_size)
        print(f"Chunking completed! Generated {len(corpus_chunks)} total chunks.")
    except Exception as e:
        print(f"[FATAL ERROR] Failed to chunk data: {e}")
        sys.exit(1)
        
    # Save to CSV
    try:
        df_chunks = pd.DataFrame(corpus_chunks)
        df_chunks.to_csv(csv_path, index=False)
        print(f"Successfully saved chunk corpus to: {csv_path}")
    except Exception as e:
        print(f"[FATAL ERROR] Failed to save chunk corpus: {e}")
        sys.exit(1)
        
    # Download/Verify local model cache
    print("\n--- Phase 3: Verifying/Downloading Retriever Models ---")
    try:
        download_retriever_models()
    except Exception as e:
        print(f"[FATAL ERROR] Failed to download retriever models: {e}")
        sys.exit(1)
        
    print("\n=== Bootstrap Setup Complete for RAG-2/3! ===")

if __name__ == "__main__":
    main()
