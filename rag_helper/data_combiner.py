import os
import glob
import pandas as pd
from pathlib import Path

def get_combined_dataframe(parsing_dir: Path) -> pd.DataFrame:
    """
    Finds and reads all parsed Excel files under parsing_dir and combines them into one DataFrame.
    Each row is annotated with the 'File Name' of the source PDF.
    """
    search_pattern = str(parsing_dir / "*" / "*.xlsx")
    xlsx_paths = glob.glob(search_pattern)
    
    if not xlsx_paths:
        raise FileNotFoundError(f"No Excel files found matching: {search_pattern}")
        
    print(f"Found {len(xlsx_paths)} Excel file(s) under '{parsing_dir}'. Combining...")
    
    dfs = []
    for path_str in sorted(xlsx_paths):
        path = Path(path_str)
        pdf_name = path.parent.name  # e.g., 'Agrani Bank.pdf'
        
        try:
            df = pd.read_excel(path)
            if df.empty:
                print(f"[WARNING] File '{path.name}' is empty. Skipping.")
                continue
                
            # Add metadata columns
            df["File Name"] = pdf_name
            
            # Standardize sheet columns (strip spaces in headers if any)
            df.columns = [col.strip() for col in df.columns]
            
            dfs.append(df)
            print(f" - Loaded '{pdf_name}' ({len(df)} rows)")
        except Exception as e:
            print(f"[ERROR] Failed to read '{path}': {e}")
            
    if not dfs:
        raise ValueError("No data could be successfully loaded from the Excel files.")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df
