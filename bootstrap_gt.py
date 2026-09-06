import os
import json
import argparse
from pathlib import Path
import pandas as pd
from utils.qa_helper import execute_query

ROOT_DIR = Path(__file__).parent.resolve()

def load_questions():
    dataset_path = ROOT_DIR / "data" / "QA" / "dataset.json"
    if not dataset_path.exists():
        print(f"[ERROR] QA Dataset not found at '{dataset_path}'. Please run bootstrap first.")
        import sys
        sys.exit(1)
        
    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Keep a positional q_num for display only; the canonical key is the stable `id`.
    for idx, q in enumerate(questions, 1):
        q["q_num"] = idx

    return questions

def main():
    parser = argparse.ArgumentParser(description="Run ground truth SQL queries and save to Excel (keyed by question id).")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of questions to process.")
    parser.add_argument("--force", action="store_true", help="Regenerate even if the output file already exists.")
    args = parser.parse_args()

    questions = load_questions()
    if args.limit is not None:
        questions = questions[:args.limit]

    output_dir = ROOT_DIR / "output" / "qa" / "evaluation" / "ground_truth"
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(questions)
    print(f"Starting ground truth SQL execution for {total} questions...")

    for i, q in enumerate(questions, 1):
        # Ground-truth artifacts are keyed by the stable question `id` (e.g. A_017), NOT a
        # positional number, so deleting/adding items never shifts the file<->item mapping.
        q_id = q["id"]
        excel_path = output_dir / f"{q_id}.xlsx"

        # Skip if already exists (pause-resume capability)
        if excel_path.exists() and not args.force:
            print(f"[{i}/{total}] {q_id}: already exists. Skipping.")
            continue

        sql = q.get("ground_truth_sql")
        if not sql:
            print(f"[{i}/{total}] {q_id}: WARNING no ground_truth_sql. Skipping.")
            continue

        print(f"[{i}/{total}] Executing SQL for question ID: {q_id}")
        try:
            results = execute_query(sql)
            with pd.ExcelWriter(str(excel_path), engine="openpyxl") as writer:
                if results:
                    df = pd.DataFrame(results)
                    # Convert timezone-aware datetimes to timezone-unaware for Excel compatibility
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
            print(f"  Saved to {excel_path.name}")
        except Exception as e:
            print(f"  ERROR executing/saving {q_id}: {e}")

if __name__ == "__main__":
    main()
