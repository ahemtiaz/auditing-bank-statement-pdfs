import os
import sys
import json
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

# Setup paths
ROOT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(ROOT_DIR))

# Import enhanced SQLAgent and DB helpers
from sql_agent import SQLAgent
from utils.qa_helper import execute_query, get_db_schema
from rag_helper import DocumentProvider, SentenceTransformerRetriever

def main():
    parser = argparse.ArgumentParser(description="Interactive single-question runner.")
    parser.add_argument(
        "--solution", 
        default="zero_context_sql",
        choices=[
            "zero_context_sql", 
            "self_correcting_sql", 
            "unstructured_rag_sql", 
            "structured_rag_sql", 
            "self_correcting_structured_rag_sql"
        ],
        help="The solution pipeline to use."
    )
    parser.add_argument(
        "--question", 
        required=True,
        help="The user's question to answer."
    )
    args = parser.parse_args()
    
    load_dotenv()
    
    solution_name = args.solution
    question_text = args.question
    
    print("\n=================================================================")
    print(f"      Interactive Query Runner | Solution: {solution_name}")
    print("=================================================================")
    print(f"Question: '{question_text}'")
    
    is_rag = "rag" in solution_name
    use_validator = "self_correcting" in solution_name
    
    # Configure retriever if RAG
    retrieved_context = None
    k_retrieve = 1 if "self_correcting" in solution_name else 3
    
    if is_rag:
        # Determine RAG corpus CSV
        if "unstructured" in solution_name:
            csv_path = ROOT_DIR / "output" / "parsing" / "unstructured_rag" / "chunk_corpus.csv"
        else:
            csv_path = ROOT_DIR / "output" / "parsing" / "structured_rag" / "chunk_corpus.csv"
            
        if not csv_path.exists():
            print(f"[ERROR] RAG chunk corpus not found at '{csv_path}'. Please run bootstrap first.")
            sys.exit(1)
            
        print(f"Loading retriever from: {csv_path.name}...")
        model_to_load = "BAAI/bge-m3"
        
        try:
            provider = DocumentProvider(csv_path)
            retriever = SentenceTransformerRetriever(provider, model_name=model_to_load)
            
            print(f"Retrieving top-{k_retrieve} chunks...")
            q_emb = retriever.model.encode([question_text], normalize_embeddings=True, show_progress_bar=False).astype("float32")
            D, I = retriever.index.search(q_emb, k=k_retrieve)
            
            retrieved = []
            for idx in I[0]:
                chunk_text = provider.texts[idx]
                retrieved.append(chunk_text)
                
            if retrieved:
                retrieved_context = "\n\n".join([f"--- Chunk {i} ---\n{text}" for i, text in enumerate(retrieved, 1)])
                print("Retrieval completed successfully.")
            else:
                print("WARNING: No chunks retrieved.")
        except Exception as e:
            print(f"[ERROR] Failed to run retrieval: {e}")
            sys.exit(1)

    domain_context = os.getenv(
        "DATASET_DOMAIN_CONTEXT",
        "The dataset consists of Bangladeshi bank statements. Unless the question states otherwise, "
        "assume Bangladesh conventions throughout (e.g. the weekend is Friday and Saturday; the fiscal "
        "year ends 30 June; the currency is BDT."
    )
    # Initialize SQL Agent
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    agent = SQLAgent(
        db_schema=get_db_schema(),
        executor_fn=execute_query,
        model_name=gemini_model,
        max_attempts=5,
        min_delay_seconds=1.5,
        use_validator=use_validator,
        domain_context=domain_context
    )
    
    start_time = time.time()
    try:
        response = agent.run(question_text, retrieved_context=retrieved_context)
        elapsed_ms = (time.time() - start_time) * 1000
        
        query = response.get("query", "")
        results = response.get("results", [])
        total_count = response.get("total_count", 0)
        summary = response.get("summary", "")
        metadata = response.get("metadata", {})
        error = metadata.get("error") or response.get("error")
        gen_error = metadata.get("generation_error") or response.get("generation_error")
        
        # Display to console
        print("\n--- RESULTS ---")
        print(f"Generated SQL Query:\n{query}\n")
        if error:
            print(f"Execution Error: {error}")
        if gen_error:
            print(f"Generation Error: {gen_error}")
        
        print(f"Total Rows Matched in DB: {total_count}")
        print(f"Execution Time: {elapsed_ms:.2f} ms")
        print(f"Validator Summary:\n{summary}\n")
        
        if results:
            print("Preview of results (first 5 rows):")
            df_preview = pd.DataFrame(results).head(5)
            print(df_preview.to_string(index=False))
        else:
            print("No query results returned.")
            
        # Save output files to output/qa/run/<solution>/run_<timestamp>/
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT_DIR / "output" / "qa" / "run" / solution_name / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        excel_path = output_dir / "table.xlsx"
        others_path = output_dir / "others.txt"
        
        with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
            if results:
                df = pd.DataFrame(results)
                for col in df.columns:
                    if pd.api.types.is_datetime64tz_dtype(df[col]):
                        df[col] = df[col].dt.tz_localize(None)
                    else:
                        df[col] = df[col].apply(lambda x: x.replace(tzinfo=None) if hasattr(x, 'tzinfo') and x.tzinfo is not None else x)
                df.to_excel(writer, sheet_name="query_results", index=False)
            else:
                pd.DataFrame([{"message": "No query results returned"}]).to_excel(
                    writer, sheet_name="no_results", index=False
                )
                
        with open(others_path, "w", encoding="utf-8") as f:
            f.write("[GENERATED SQL QUERY]\n")
            f.write(f"{query}\n\n")
            f.write("[METADATA]\n")
            f.write(f"execution_time_ms: {elapsed_ms}\n")
            for k, v in metadata.items():
                f.write(f"{k}: {v}\n")
            if error:
                f.write(f"\n[EXECUTION ERROR]\n{error}\n")
            if gen_error:
                f.write(f"\n[GENERATION ERROR]\n{gen_error}\n")
                
        print(f"\n[SUCCESS] Interactive output saved to: {output_dir}")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline run crashed: {e}")

if __name__ == "__main__":
    main()
