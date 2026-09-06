import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .llm_client import GeminiLLMClient


logger = logging.getLogger(__name__)

class ValidationResult(BaseModel):
    approved: bool = Field(description="True if the generated SQL query is fully correct, secure, and execution results look logically correct and answer the user question.")
    error_message: str = Field(description="If not approved, a detailed constructive feedback message explaining why and instructing how to fix it.")
    corrected_query: Optional[str] = Field(None, description="Optional. If you can fix the query directly, provide the complete updated corrected SELECT SQL query here.")
    analysis_summary: Optional[str] = Field(None, description="A comprehensive, detailed natural language summary of the results. This is required if approved is True or if this is the final iteration.")


class SQLValidator:
    """Validates generated SQL queries and their execution output against the user question."""
    def __init__(self, db_schema: str):
        self.db_schema = db_schema

    def validate(
        self,
        client: GeminiLLMClient,
        user_query: str,
        query: str,
        execution_rows: List[Dict[str, Any]],
        total_count: int,
        is_final: bool = False,
        generator_history: Optional[List[Dict[str, Any]]] = None,
        validator_history: Optional[List[Dict[str, Any]]] = None,
        generator_reasoning: Optional[str] = None,
        retrieved_context: Optional[str] = None,
        domain_context: Optional[str] = None
    ) -> ValidationResult:
        """Invokes Gemini to evaluate the query logic, safety, and returned database rows."""
        generator_history = generator_history or []
        validator_history = validator_history or []
        
        clean_query = query.strip().rstrip(';')
        count_query = f"SELECT COUNT(*) FROM (\n{clean_query}\n) AS count_subquery"

        # Prepare context of generated query and execution preview
        queries_context = f"\n--- Query ---\n"
        queries_context += f"Primary SQL Statement:\n{query}\n"
        queries_context += f"Companion Count Statement:\n{count_query}\n"
        queries_context += f"Total Matching Rows in DB: {total_count}\n"
        queries_context += "Execution Preview (First few rows):\n"
        if execution_rows:
            queries_context += json.dumps(execution_rows, indent=2, default=str) + "\n"
        else:
            queries_context += "  (No rows returned/Empty dataset)\n"

        schema_part = f"""You are a strict, expert Financial Auditor and SQL Verification Analyst.
Your role is to audit generated SQL queries and their actual database execution results to verify they are perfectly secure, syntactically correct, logically sound, and fully answer the user's natural language question.

=== DATABASE SCHEMA ===
The database has a single table named `transactions` representing bank statements.
Schema definition:
{self.db_schema}
"""

        prompt = f"""{schema_part}

=== AUDIT INPUTS ===
Original User Question: {user_query}
Is Final Attempt: {is_final}
"""

        if domain_context:
            prompt += (
                "\n=== DATASET CONTEXT & ASSUMPTIONS ===\n"
                f"{domain_context}\n"
                "Apply these assumptions only where the question itself does not specify otherwise.\n"
            )

        if retrieved_context:
            prompt += f"""
=== RETRIEVED STATEMENT CONTEXT (RAG) ===
Use this raw text retrieved from bank statement files to guide your understanding. This is NOT the output, but a reference to check the database execution results against:
{retrieved_context}
"""


        if generator_reasoning:
            prompt += f"Generator Proposed Reasoning / Plan: {generator_reasoning}\n"

        # Append generator history if not empty
        if generator_history:
            prompt += "\n=== PREVIOUS GENERATION AND EXECUTION ATTEMPTS ===\n"
            for i, h in enumerate(generator_history):
                prompt += f"\nAttempt {i+1}:\n"
                if "query" in h and h["query"]:
                    prompt += f"  Proposed Query:\n    {h.get('query')}\n"
                if "errors" in h and h["errors"]:
                    prompt += "  Errors Encountered:\n"
                    for err in h["errors"]:
                        prompt += f"    - {err}\n"

        # Append validator history if not empty
        if validator_history:
            prompt += "\n=== PREVIOUS VALIDATION AUDITS ===\n"
            for i, h in enumerate(validator_history):
                prompt += f"\nValidation Loop {i+1}:\n"
                if "query" in h and h["query"]:
                    prompt += f"  Audited (Rejected) Query:\n    {h.get('query')}\n"
                if "issue" in h and h["issue"]:
                    prompt += f"  Validator Rejection Feedback: {h['issue']}\n"

        prompt += f"""
=== GENERATED QUERY & EXECUTION RESULTS ===
{queries_context}

=== AUDITING PROTOCOL ===
1. SECURITY: Reject any query that is not a pure read-only SELECT statement (no UPDATE, INSERT, DELETE, DROP, ALTER, TRUNCATE, or other mutating keywords).
2. QUESTION-QUERY COVERAGE: Verify the SQL query addresses the complete user question — not a partial subset and not more than what was asked. Every condition, entity, and metric mentioned in the question should be reflected in the query, and the query should not introduce extra filters or computations the user did not request.
3. RESULT INSPECTION: Examine the returned preview rows for false positives (rows that should not match) and false negatives (expected rows that are missing). If you spot suspicious data, check whether the generated query has issues such as: improper NULL handling (e.g., missing COALESCE in arithmetic), overly strict filtering (e.g., exact match where substring was needed), overly loose filtering (e.g., broad LIKE matching short abbreviations), incorrect column references, or wrong date/value boundaries. Only reject if you can identify a concrete query defect causing the bad results.
4. ZERO-RESULT HANDLING: If the query returns zero rows, inspect the question and query logic carefully. However, do NOT over-engineer — zero results can be legitimately correct for some questions. Only reject if there is clear evidence (from the question phrasing or RAG context) that matching data should exist and the query has a specific bug preventing it.




=== RESPONSE & SYNTHESIS RULES ===
- If the query is incorrect, unsafe, or returns logically flawed results, set `approved = False`. Provide a detailed explanation in `error_message` explaining what was wrong and how to fix it.
- **IMPORTANT**: If `approved = False`, you MUST attempt to fix the query directly and provide the updated corrected SQL query in `corrected_query`, with all column references strictly matching the database schema.
- If the query is fully correct and secure, set `approved = True` and generate a comprehensive `analysis_summary`.
- If `is_final` is True, you MUST generate the `analysis_summary` even if you do not approve the query (synthesize the best possible answer from what was returned, explaining any database errors/issues transparently).

=== ANALYSIS SUMMARY GUIDELINES ===
- Ground strictly in the returned rows/counts; never hallucinate.
- No meta-commentary: do not mention SQL, queries, tables, the database, or "preview/results". Write a direct, UI-ready answer to the question.
- If 'Total Matching Rows in DB' > 20, the preview is partial — say so; don't claim the full set is shown.
- Be clear and professional: format numbers with commas (e.g. 10,000,000) and name relevant banks/accounts/dates.

Perform your audit now. Check safety, SQL logic mapping, date handling, null protection, and result correctness.
If approved is False, explain what needs correction and provide the corrected query. If approved is True (or is_final is True), generate the final rich analytical summary.
"""

        logger.debug(f"Validator Prompt:\n{prompt}")
        return client.generate_structured_output(prompt, ValidationResult, temperature=0.0)
