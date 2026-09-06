import os
import sys
import json
import time
import argparse
import decimal
import datetime
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Adjust path to find modules
ROOT_DIR = Path(__file__).parent.resolve()
sys.path.append(str(ROOT_DIR))

# Import enhanced SQLAgent and DB helpers
from sql_agent import SQLAgent
from utils.qa_helper import execute_query, get_db_schema
from rag_helper import DocumentProvider, SentenceTransformerRetriever
from eval_metric import compare_results, calculate_metrics, normalize_dataframe, clean_cell_value, parse_primary_order_key  # noqa: F401



# -----------------------------------------------------------------------------
# QA dataset loading
# -----------------------------------------------------------------------------
def load_dataset():
    dataset_path = ROOT_DIR / "data" / "QA" / "dataset.json"
    if not dataset_path.exists():
        print(f"[ERROR] QA Dataset not found at '{dataset_path}'. Please run bootstrap first.")
        sys.exit(1)
    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    # Assign index (1-indexed)
    for idx, q in enumerate(questions, 1):
        q["q_num"] = idx
    return questions

# -----------------------------------------------------------------------------
# Main Evaluation Orchestration
# -----------------------------------------------------------------------------
def run_solution_qa(solution_name, questions):
    print(f"\n=================================================================")
    print(f"Executing QA Pipeline for Solution: '{solution_name}'")
    print(f"=================================================================")
    
    # Configuration based on solution
    is_rag = "rag" in solution_name
    use_validator = "self_correcting" in solution_name
    
    # Configure retriever if RAG
    retriever = None
    provider = None
    k_retrieve = 3  # Default k for the two single-shot RAG solutions (structured + unstructured)
    
    if is_rag:
        # Determine RAG corpus CSV
        if "unstructured" in solution_name:
            csv_path = ROOT_DIR / "output" / "parsing" / "unstructured_rag" / "chunk_corpus.csv"
        else:
            csv_path = ROOT_DIR / "output" / "parsing" / "structured_rag" / "chunk_corpus.csv"
            
        if not csv_path.exists():
            print(f"[ERROR] RAG chunk corpus not found at '{csv_path}'. Please run bootstrap first.")
            return False
            
        print(f"Loading retriever for corpus: {csv_path.name}")
        model_to_load = "BAAI/bge-m3"
        
        try:
            provider = DocumentProvider(csv_path)
            retriever = SentenceTransformerRetriever(provider, model_name=model_to_load)
        except Exception as e:
            print(f"[ERROR] Failed to initialize RAG retriever: {e}")
            return False
            

    # Settable dataset/locale assumptions passed to the agent as a parameter (NOT hardcoded in the
    # agent or its prompts). Override per corpus via the DATASET_DOMAIN_CONTEXT env var.
    domain_context = os.getenv(
        "DATASET_DOMAIN_CONTEXT",
        "The dataset consists of Bangladeshi bank statements. Unless the question states otherwise, "
        "assume Bangladesh conventions throughout (e.g. the weekend is Friday and Saturday; the fiscal "
        "year ends 30 June; the currency is BDT."
    )

    # Initialize SQL Agent (generic; no corpus-specific logic baked in).
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
    
    output_root = ROOT_DIR / "output" / "qa" / "evaluation" / solution_name
    
    for q in questions:
        q_num = q["q_num"]
        # Prediction artifacts are keyed by the stable question `id` (e.g. A_017), not a positional
        # index, so deletions/additions never shift the file<->item mapping.
        q_dir = output_root / q["id"]
        excel_path = q_dir / "table.xlsx"
        others_path = q_dir / "others.txt"

        # Pause-resume: skip if result exists
        if excel_path.exists():
            print(f"[{q_num}/{len(questions)}] {q['id']}: Skipping: already executed.")
            continue
            
        q_dir.mkdir(parents=True, exist_ok=True)
        question_text = q.get("question")
        
        # Build custom question with schema instruction
        must_present = q.get("list_of_column_must_be_present", [])
        column_property = q.get("column_property", [])
        column_instruction = (
            f"###column instruction###\n"
            f"Columns to return (in this exact sequence and order):\n"
            f"{json.dumps(must_present, ensure_ascii=False)}\n\n"
            f"Column Properties (for new/synthetic/computed columns):\n"
            f"{json.dumps(column_property, indent=2, ensure_ascii=False)}\n\n"
            f"Instructions:\n"
            f"1. Your final response (SQL query results) column names, order, and count must match the above list of columns exactly. Do not return any extra columns, and do not omit any requested columns.\n"
            f"2. For any new or synthetic column, the values returned in the query cells must match the properties specified to be considered correct."
        )
        custom_question = f"###question###\n{question_text}\n\n{column_instruction}"
        
        print(f"[{q_num}/{len(questions)}] Running query: '{question_text}'")
        
        # Run retrieval if RAG
        retrieved_context = None
        if is_rag and retriever and provider:
            try:
                q_emb = retriever.model.encode([question_text], normalize_embeddings=True, show_progress_bar=False).astype("float32")
                D, I = retriever.index.search(q_emb, k=k_retrieve)
                
                retrieved = []
                for idx in I[0]:
                    chunk_text = provider.texts[idx]
                    retrieved.append(chunk_text)
                    
                if retrieved:
                    retrieved_context = "\n\n".join([f"--- Chunk {i} ---\n{text}" for i, text in enumerate(retrieved, 1)])
            except Exception as e:
                print(f"  Retrieval error: {e}")
                
        # Call agent. The column contract is already embedded in custom_question; the agent is
        # expected to follow it (and is penalized by the metric if it does not).
        try:
            response = agent.run(custom_question, retrieved_context=retrieved_context)
            
            results = response.get("results", [])
            query = response.get("query", "")
            reasoning = response.get("reasoning", "No reasoning generated.")
            summary = response.get("summary", "")
            metadata = response.get("metadata", {})
            error = metadata.get("error") or response.get("error")
            gen_error = metadata.get("generation_error") or response.get("generation_error")
            
            # Save results to Excel
            with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
                if results:
                    df = pd.DataFrame(results)
                    # Safe local datetime conversions
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
                    
            # Save logs to others.txt
            with open(others_path, "w", encoding="utf-8") as f:
                f.write("[GENERATED SQL QUERY]\n")
                f.write(f"{query}\n\n")
                f.write("[REASONING]\n")
                f.write(f"{reasoning}\n\n")
                f.write("[VALIDATOR SUMMARY]\n")
                f.write(f"{summary}\n\n")
                f.write("[METADATA]\n")
                for k, v in metadata.items():
                    f.write(f"{k}: {v}\n")
                if error:
                    f.write(f"\n[EXECUTION ERROR]\n{error}\n")
                if gen_error:
                    f.write(f"\n[GENERATION ERROR]\n{gen_error}\n")
            print(f"  Completed successfully.")
        except Exception as e:
            print(f"  Execution exception: {e}")
            with open(others_path, "w", encoding="utf-8") as f:
                f.write("[ERROR]\n")
                f.write(f"Pipeline crashed: {e}\n")
                
    # Run evaluation against ground truth
    print(f"\nEvaluating execution results for '{solution_name}'...")
    q_results = []
    
    for q in questions:
        q_num = q["q_num"]
        q_id = q["id"]
        order_matters = q.get("order_matter_or_not", False)
        question_text = q.get("question")
        df_pred = _read_result_excel(_pred_path(output_root, q_id, q_num))
        df_true = _read_result_excel(_gt_path(q_id, q_num))

        order_key = parse_primary_order_key(q.get("ground_truth_sql", "")) if order_matters else None
        tp, fp, fn, order_ok = compare_results(df_pred, df_true, order_matters, order_key)
        precision, recall, f1, jaccard = calculate_metrics(tp, fp, fn)
        set_match = int(fp == 0 and fn == 0)               # content identical as a multiset
        exec_accuracy = int(set_match == 1 and order_ok)   # PRIMARY: content + ties-tolerant order

        q_results.append({
            "q_num": q_num,
            "question_id": q_id,
            "question": question_text,
            "order_matter": bool(order_matters),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "set_precision": precision,
            "set_recall": recall,
            "set_f1": f1,
            "set_jaccard": jaccard,
            "order_satisfied": int(order_ok),
            "execution_accuracy": exec_accuracy,   # primary headline metric
            "exact_match": exec_accuracy,          # demoted strict secondary (== exec_acc here)
        })

    df_q_metrics = pd.DataFrame(q_results)
    if len(df_q_metrics):
        print(f"  OVERALL: execution_accuracy={df_q_metrics['execution_accuracy'].mean():.3f}  "
              f"set_f1={df_q_metrics['set_f1'].mean():.3f}  (n={len(df_q_metrics)})")

    # Save/Append to the single consolidated evaluation excel (one sheet per solution).
    eval_excel_path = ROOT_DIR / "output" / "qa" / "evaluation" / "evaluation_results.xlsx"
    eval_excel_path.parent.mkdir(parents=True, exist_ok=True)

    sheets_dict = {}
    if eval_excel_path.exists():
        try:
            with pd.ExcelFile(eval_excel_path) as xls:
                for sheet in xls.sheet_names:
                    sheets_dict[sheet] = pd.read_excel(xls, sheet_name=sheet)
        except Exception as e:
            print(f"[WARNING] Could not read existing evaluation file: {e}")

    sheets_dict[solution_name] = df_q_metrics

    try:
        with pd.ExcelWriter(eval_excel_path, engine="openpyxl") as writer:
            for sheet_name, df_sheet in sheets_dict.items():
                df_sheet.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f"Saved evaluation sheets for '{solution_name}' in: {eval_excel_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save consolidated evaluation sheets: {e}")

    return True


def _read_result_excel(path):
    """Read a result Excel, preferring the 'query_results' sheet; None if absent/unreadable."""
    if path is None or not path.exists():
        return None
    try:
        return pd.read_excel(path, sheet_name="query_results")
    except Exception:
        try:
            return pd.read_excel(path, sheet_name=0)
        except Exception:
            return None


def _pred_path(output_root, q_id, q_num):
    """Prediction path: id-keyed dir preferred; fall back to legacy positional dir."""
    p = output_root / q_id / "table.xlsx"
    if p.exists():
        return p
    legacy = output_root / str(q_num) / "table.xlsx"
    return legacy if legacy.exists() else p


def _gt_path(q_id, q_num):
    """Ground-truth path: id-keyed preferred; fall back to legacy positional file."""
    base = ROOT_DIR / "output" / "qa" / "evaluation" / "ground_truth"
    p = base / f"{q_id}.xlsx"
    if p.exists():
        return p
    legacy = base / f"{q_num}.xlsx"
    return legacy if legacy.exists() else p

# -----------------------------------------------------------------------------
# Main CLI Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Consolidated QA runner and evaluator.")
    parser.add_argument(
        "--solution", 
        choices=[
            "zero_context_sql", 
            "unstructured_rag_sql", 
            "structured_rag_sql", 
            "self_correcting_sql", 
            "self_correcting_structured_rag_sql"
        ],
        default=None, 
        help="Specify which solution to execute and evaluate. If omitted, runs all 5."
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit the number of questions to process for testing."
    )
    args = parser.parse_args()
    
    load_dotenv()
    questions = load_dataset()
    if args.limit is not None:
        questions = questions[:args.limit]
        
    solutions = [args.solution] if args.solution else [
        "zero_context_sql", 
        "structured_rag_sql", 
        "self_correcting_sql", 
        "self_correcting_structured_rag_sql",
        "unstructured_rag_sql"
    ]
    
    print("=================================================================")
    print("         Unified Bank Statement QA Execution & Evaluation        ")
    print("=================================================================")
    print(f"Processing {len(questions)} questions for solutions: {solutions}")
    
    for sol in solutions:
        success = run_solution_qa(sol, questions)
        if not success:
            print(f"[WARNING] Solution '{sol}' processing failed.")
            
    print("\nAll operations completed!")

if __name__ == "__main__":
    main()
