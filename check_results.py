import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def calculate_metrics(tp, fp, fn):
    pred_total = tp + fp
    if pred_total == 0:
        precision = 1.0 if (tp + fn) == 0 else 0.0
    else:
        precision = tp / pred_total
        
    true_total = tp + fn
    if true_total == 0:
        recall = 1.0 if pred_total == 0 else 0.0
    else:
        recall = tp / true_total
        
    if precision + recall == 0:
        f1 = 1.0 if (pred_total == 0 and true_total == 0) else 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    union_total = tp + fp + fn
    if union_total == 0:
        accuracy = 1.0
    else:
        accuracy = tp / union_total
        
    return precision, recall, f1, accuracy

def main():
    parser = argparse.ArgumentParser(description="Evaluate consolidated QA sheets and print stats.")
    parser.add_argument(
        "--save-excel", 
        action="store_true", 
        help="If supplied, saves the aggregated summary to output/qa/aggregated_evaluation_summary.xlsx"
    )
    args = parser.parse_args()

    workspace_dir = Path(__file__).parent.resolve()
    eval_excel_path = workspace_dir / "output" / "qa" / "evaluation" / "evaluation_results.xlsx"
    summary_path = workspace_dir / "output" / "qa" / "evaluation" / "aggregated_evaluation_summary.xlsx"

    if not eval_excel_path.exists():
        print(f"[ERROR] Evaluation file not found at '{eval_excel_path}'.")
        print("Please run `python run_eval.py` first to generate evaluations.")
        return

    print("=================================================================")
    print("               Bank Statement QA Status and Metrics              ")
    print("=================================================================")
    
    try:
        xls = pd.ExcelFile(eval_excel_path)
    except Exception as e:
        print(f"[ERROR] Failed to load '{eval_excel_path}': {e}")
        return

    summary_rows = []
    
    # Load dataset.json to dynamically count total questions
    dataset_path = workspace_dir / "data" / "QA" / "dataset.json"
    total_questions = 0
    if dataset_path.exists():
        try:
            import json
            with open(dataset_path, "r", encoding="utf-8") as f:
                total_questions = len(json.load(f))
        except Exception as e:
            print(f"[WARNING] Could not parse dataset.json: {e}")

    for sheet_name in sorted(xls.sheet_names):
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        except Exception as e:
            print(f"Failed to read sheet '{sheet_name}': {e}")
            continue
            
        processed_count = len(df)
        # Use dynamic total count if available; fallback to current processed count
        sheet_total_questions = total_questions if total_questions > 0 else processed_count
        pending_count = max(0, sheet_total_questions - processed_count)
        
        # Calculate Macro averages
        macro_pr = df["set_precision"].mean() if "set_precision" in df.columns else (df["precision"].mean() if "precision" in df.columns else 0.0)
        macro_re = df["set_recall"].mean() if "set_recall" in df.columns else (df["recall"].mean() if "recall" in df.columns else 0.0)
        macro_f1 = df["set_f1"].mean() if "set_f1" in df.columns else (df["f1"].mean() if "f1" in df.columns else 0.0)
        macro_acc = df["set_jaccard"].mean() if "set_jaccard" in df.columns else (df["accuracy"].mean() if "accuracy" in df.columns else 0.0)
        
        # Calculate Execution Accuracy (Exact Match %)
        perfect_matches = df["exact_match"].sum() if "exact_match" in df.columns else 0
        execution_accuracy_pct = (perfect_matches / sheet_total_questions) * 100 if sheet_total_questions > 0 else 0.0
        
        print(f"\nSolution: '{sheet_name}'")
        print(f"  - Progress: {processed_count}/{sheet_total_questions} queries run ({pending_count} pending)")
        if processed_count > 0:
            print(f"  - Execution Accuracy (Exact Match): {execution_accuracy_pct:.2f}% ({perfect_matches}/{sheet_total_questions})")
            print(f"  - Macro Average: F1: {macro_f1*100:.2f}% | Jaccard: {macro_acc*100:.2f}% | Precision: {macro_pr*100:.2f}% | Recall: {macro_re*100:.2f}%")
        
        summary_rows.append({
            "Solution": sheet_name,
            "Queries Processed": processed_count,
            "Queries Pending": pending_count,
            "Execution Accuracy (Exact Match %)": execution_accuracy_pct,
            "Macro F1 (%)": macro_f1 * 100,
            "Macro Jaccard Accuracy (%)": macro_acc * 100,
            "Macro Precision (%)": macro_pr * 100,
            "Macro Recall (%)": macro_re * 100
        })

    if args.save_excel and summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        try:
            df_summary.to_excel(summary_path, index=False)
            print(f"\n[SUCCESS] Aggregated summary successfully saved to: {summary_path}")
        except Exception as e:
            print(f"\n[ERROR] Failed to save summary file: {e}")

if __name__ == "__main__":
    main()
