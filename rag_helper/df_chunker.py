import io
import csv
import pandas as pd
import tiktoken
from pathlib import Path
from typing import List, Dict, Any

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Helper to count the number of tokens in a text string using tiktoken."""
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        encoding = tiktoken.get_encoding("gpt2")
    return len(encoding.encode(text))

def list_to_csv_string(rows: List[Dict[str, Any]], fieldnames: List[str]) -> str:
    """Converts a list of row dictionaries to a CSV formatted string with the specified header fields."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()

def chunk_dataframe_to_csv(
    df: pd.DataFrame, 
    max_tokens: int = 512, 
    encoding_name: str = "cl100k_base"
) -> List[Dict[str, Any]]:
    """
    Chunks a combined DataFrame containing transactions from multiple bank statements.
    
    Constraints:
    1. Output is formatted as a CSV.
    2. Each chunk starts with the header row.
    3. No overlap between chunks.
    4. One single transaction row is in a single chunk (no splitting transactions).
    5. Accommodate as many rows as possible dynamically up to max_tokens.
    6. For each statement, completely null/NaN columns are dropped to save tokens.
    7. Groups by Page to preserve exact page association for dense retrieval and evaluation.
    
    Returns:
        List of dicts containing chunk metadata:
        [
            {
                "chunk_id": str,
                "text_description": str (the CSV string),
                "image_filename": str,
                "query": str
            },
            ...
        ]
    """
    corpus_chunks = []
    
    # 1. Group by file name. Each parsed PDF statement is processed independently.
    file_column = "File Name" if "File Name" in df.columns else "file_name"
    if file_column not in df.columns:
        df[file_column] = "unknown_statement.pdf"
        
    for file_name, file_group in df.groupby(file_column):
        file_stem = Path(file_name).stem
        
        # 2. Clean the statement group: drop columns that are entirely NaN/null for this statement.
        cleaned_file_df = file_group.dropna(how="all", axis=1)
        
        # Fill remaining NaN values with empty string so they serialize as empty fields in CSV
        # (avoiding printing "nan" or "None" in CSV output)
        cleaned_file_df = cleaned_file_df.fillna("")
        
        # 3. Group by Page within the statement
        page_column = "Page" if "Page" in cleaned_file_df.columns else "page"
        if page_column not in cleaned_file_df.columns:
            cleaned_file_df[page_column] = 1
            
        for page_num, page_group in cleaned_file_df.groupby(page_column):
            # Sort page transactions by 'Serial' or row index to preserve order
            sort_cols = [col for col in ["Serial", "serial", "Date", "date"] if col in page_group.columns]
            if sort_cols:
                page_group = page_group.sort_values(by=sort_cols)
                
            # Columns to include in this bank's CSV header
            fieldnames = [col for col in page_group.columns if col not in {file_column}]
            
            rows = page_group[fieldnames].to_dict(orient="records")
            
            current_chunk_rows = []
            chunk_idx = 1
            
            for row in rows:
                # Test adding this row to the current chunk
                test_rows = current_chunk_rows + [row]
                test_csv = list_to_csv_string(test_rows, fieldnames)
                test_tokens = count_tokens(test_csv, encoding_name)
                
                if test_tokens > max_tokens:
                    # Exceeded limit.
                    if current_chunk_rows:
                        # Save the current chunk first
                        chunk_csv = list_to_csv_string(current_chunk_rows, fieldnames)
                        chunk_id = f"{file_stem}_page_{page_num}_chunk_{chunk_idx}"
                        image_filename = f"{file_stem}_page_{page_num}.png"
                        
                        corpus_chunks.append({
                            "chunk_id": chunk_id,
                            "text_description": chunk_csv,
                            "image_filename": image_filename,
                            "query": ""
                        })
                        chunk_idx += 1
                        
                        # Start new chunk with the current row
                        current_chunk_rows = [row]
                    else:
                        # A single row alone exceeds max_tokens. 
                        # We must emit it in its own chunk to avoid transaction split.
                        chunk_id = f"{file_stem}_page_{page_num}_chunk_{chunk_idx}"
                        image_filename = f"{file_stem}_page_{page_num}.png"
                        
                        corpus_chunks.append({
                            "chunk_id": chunk_id,
                            "text_description": test_csv,
                            "image_filename": image_filename,
                            "query": ""
                        })
                        chunk_idx += 1
                        current_chunk_rows = []
                else:
                    # Fits in current chunk
                    current_chunk_rows.append(row)
            
            # Emit any remaining rows for this page
            if current_chunk_rows:
                chunk_csv = list_to_csv_string(current_chunk_rows, fieldnames)
                chunk_id = f"{file_stem}_page_{page_num}_chunk_{chunk_idx}"
                image_filename = f"{file_stem}_page_{page_num}.png"
                
                corpus_chunks.append({
                    "chunk_id": chunk_id,
                    "text_description": chunk_csv,
                    "image_filename": image_filename,
                    "query": ""
                })
                
    return corpus_chunks
