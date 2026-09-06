import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .llm_client import GeminiLLMClient

logger = logging.getLogger(__name__)

class GeneratedQuery(BaseModel):
    reasoning: str = Field(description="Chain-of-thought analysis explaining the SQL logic and how the CTEs/subqueries are structured to address the user question in a single query.")
    query: str = Field(description="The primary SELECT SQL query. Clean, standardized, and uncapped (no LIMIT unless explicitly requested by the user).")

    @property
    def count_query(self) -> str:
        """Deterministically generates the corresponding COUNT query from the primary query."""
        clean_query = self.query.strip().rstrip(';')
        return f"SELECT COUNT(*) FROM (\n{clean_query}\n) AS count_subquery"


class SQLGenerator:
    """Generates precise SQL queries based on database schema, guidelines, retrieved context, and feedback history."""
    def __init__(self, db_schema: str):
        self.db_schema = db_schema

    def generate_query(
        self,
        client: GeminiLLMClient,
        user_query: str,
        generator_history: List[Dict[str, Any]] = None,
        validator_history: List[Dict[str, Any]] = None,
        retrieved_context: Optional[str] = None,
        domain_context: Optional[str] = None
    ) -> GeneratedQuery:
        """Formulates the system prompt and requests Gemini to generate the query."""
        generator_history = generator_history or []
        validator_history = validator_history or []
        
        # Build prompt instructions
        schema_part = f"""You are an expert SQL generation assistant for a PostgreSQL database.
Translate user questions into a single precise, read-only SELECT SQL query.
You must strictly follow the rules below to ensure extreme ACCURACY, safety, low latency, and low cost.

=== DATABASE SCHEMA ===
The database has a single table named `transactions` representing bank statements.
Schema definition:
{self.db_schema}
"""

        # Caller-supplied dataset context / locale assumptions (e.g. jurisdiction-specific weekend,
        # fiscal year, currency). Passed in as a parameter -- NOT hardcoded here -- so the agent stays
        # generic across corpora.
        domain_part = ""
        if domain_context:
            domain_part = (
                "\n=== DATASET CONTEXT & ASSUMPTIONS ===\n"
                f"{domain_context}\n"
                "Apply these assumptions only where the question itself does not specify otherwise.\n"
            )

        context_part = ""
        if retrieved_context:
            context_part = f"""
=== RETRIEVED CONTEXT FROM BANK STATEMENT FILES (RAG) ===
Use the following retrieved chunks from the bank statement files to guide your understanding of transaction descriptions, bank names, accounts, date ranges, and values:
{retrieved_context}
"""

        # Input and History parts
        input_part = f"\n=== USER INPUT ===\nQuestion: {user_query}\n"

        history_part = ""
        if generator_history:
            history_part += "\n=== PREVIOUS GENERATION AND EXECUTION ATTEMPTS ===\n"
            history_part += "You have previously generated a query that either failed safety checks or database execution. Review the proposed query and its execution error logs to ensure the next generation is fully correct:\n"
            for i, h in enumerate(generator_history):
                history_part += f"\nAttempt {i+1}:\n"
                if "query" in h and h["query"]:
                    history_part += f"  Proposed Query:\n    {h.get('query')}\n"
                if "errors" in h and h["errors"]:
                    history_part += "  Errors Encountered:\n"
                    for err in h["errors"]:
                        history_part += f"    - {err}\n"

        if validator_history:
            history_part += "\n=== PREVIOUS VALIDATION AUDITS ===\n"
            history_part += "The following query successfully executed, but failed logical auditing by the validator. Review the audited query and the validator's issues/critique on why it does not answer the user question to guide your generation logic:\n"
            for i, h in enumerate(validator_history):
                history_part += f"\nValidation Loop {i+1}:\n"
                if "query" in h and h["query"]:
                    history_part += f"  Audited (Rejected) Query:\n    {h.get('query')}\n"
                if "issue" in h and h["issue"]:
                    history_part += f"  Validator Rejection Feedback: {h['issue']}\n"

        # Core SQL Rules at the end of the prompt
        rules_part = """
=== CORE SQL RULES FOR MAXIMUM ACCURACY & LATENCY ===

1. STRICT COLUMN NAME VALIDATION (PREVENTS HALLUCINATION)
    - The EXACT 23 valid database columns and their types are: id(UUID), file_name(VARCHAR), date(DATE), trans_type(VARCHAR), cheque(VARCHAR), description(TEXT), debit(NUMERIC), credit(NUMERIC), bank_name(VARCHAR), page(INTEGER), account_no(VARCHAR), passed_validation(BOOLEAN), serial(INTEGER), time(VARCHAR), value_date(DATE), reference(VARCHAR), batch_number(VARCHAR), tracer_number(VARCHAR), instrument_number(VARCHAR), trans_code(VARCHAR), branch_code(VARCHAR), branch_name(VARCHAR), search_text(TEXT).
    - Note: The `(dtype)` suffix is for your type reference only; you MUST NOT include the parenthesized data type in your SQL queries (e.g. use `date`, not `date(DATE)`).
    - You MUST ONLY use columns that exist in this list. Do NOT hallucinate standard columns.
    - Specifically, there is NO column named `transaction_code` (use `trans_code` instead), NO column named `cr_dr` (the schema has separate `debit` and `credit` columns, no credit/debit indicator column), NO column named `cheque_no` (use `cheque` instead), and NO column named `balance_type`, `balance`, or `amount` (there is no balance or amount column).
    - Correct column name is `account_no`, not `account_number`.

2. EXCLUDE INTERNAL COLUMNS FROM FINAL OUTPUT (CRITICAL)
   - You MUST NEVER return the following internal columns in the OUTERMOST SELECT clause of your primary queries: `id`, `file_name`, `page`, `passed_validation`, `search_text`.
   - These columns are for internal bookkeeping. Excluding them from the final result reduces data transmission costs and latency.
   - However, it is perfectly fine to SELECT them inside SUBQUERIES or CTEs if the outer query needs them for filtering/joining, and you CAN reference them in WHERE, JOIN, or GROUP BY clauses.

3. KEYWORD & TEXT SEARCH (GIN TRIGRAM INDEX UTILIZATION)
    - For all keyword or general text searches, search ONLY against the generated `search_text` column (lowercase & unaccented).
    - ALWAYS use substring/infix matching: `search_text LIKE '%lowercase_unaccented_keyword%'` (wildcards at both ends for better accuracy).
    - This utilizes the high-performance Trigram GIN index, reducing query latency from seconds to milliseconds.
    - NEVER use `LIKE` or `ILIKE` on `description` directly, as this causes slow table-scans.
    - Avoid `search_text = 'keyword'` or prefix/postfix matching (which fail to match long statement descriptions).
    - Avoid Broad Wildcard Matches on Short Abbreviations: If a search term or column filter is a short abbreviation or a prefix of another entity (e.g., searching for 'NRB Bank' when 'NRBC BANK' also exists, or 'Janata Bank' when 'Janata Bank PLC.' exists), do NOT use a broad substring wildcard like `ILIKE '%nrb%'` or `%janata%` directly. This will return false positive matches from other entities. Instead, include the full unique identifier (e.g. `ILIKE '%nrb bank%'` or exact match `bank_name = 'NRB Bank'`) or ensure word boundaries are matched to avoid over-matching.
    - False-positive override: `search_text LIKE '%keyword%'` is the default, but it is NOT mandatory. When a keyword is short/ambiguous or risks over-matching (recall `search_text` also bundles bank/branch/account/reference — e.g. `%lc%` hits 'PLC', `%co%` hits any word), switch to a tighter match: filter the raw field and/or use regex word-boundaries or alternation, e.g. `description ~* '\\mlc\\M'` or `search_text ~ 'reversal|write.?off|adjust|bad debt'`.

4. LINGUISTIC VOCABULARY MAPPING
    - "Inflow / Deposit / Received / Earnings / Cash In / Credit" -> maps to `credit` column.
      Always verify deposits using `credit IS NOT NULL` or `credit > 0`.
    - "Outflow / Spend / Cost / Paid / Charges / Withdrawal / Cash Out / Payment / VAT / Stamp Charge / Fee" -> maps to `debit` column.
      Always verify spending using `debit IS NOT NULL` or `debit > 0`.
    - Apply similar logical mappings for other concepts based on the schema columns (e.g., date-related terms map to `date` or `value_date`, bank identifiers to `bank_name` or `branch_code`, etc.). Verify conditions explicitly (e.g., `is not null`).

5. NULL HANDLING, DEFAULT VALUES & ARITHMETIC SAFETY
    - Arithmetic Coalescing: Apply `COALESCE(column, 0)` row-wise when performing arithmetic operations combining multiple columns (e.g. `colA - colB`), since any operation with a NULL yields NULL. However, do NOT wrap standalone aggregate functions (e.g., `SUM(column)` or `AVG(column)`) in `COALESCE` at the outermost level unless the user query explicitly requests returning a default value (e.g. "if none, return 0"). Standalone aggregates naturally handle NULLs, and wrapping them alters the query's return behavior from NULL (representing an empty set) to 0.0, which is logically distinct.
    - Inactive Columns with Default Values: When working with mutually exclusive columns (such as credit/debit, inflow/outflow, positive/negative values), do not assume that the inactive column is database NULL. It may contain a default value of 0.0, 0, or a blank space. Never use `COALESCE` to extract the active value (e.g., `COALESCE(colA, colB)`) if the inactive column can contain a default zero value, as `COALESCE` will incorrectly return the zero value instead of falling back. Instead, extract the active value using conditional logic (e.g. `CASE` statements or `GREATEST(coalesce(colA, 0), coalesce(colB, 0))` for non-negative values).
    - Division safety: Always guard against division-by-zero using `NULLIF(denominator, 0)`.
    - Floating point safety: Use numeric/float casts for accurate ratio division (e.g. `val::numeric / NULLIF(total, 0)::numeric`).

6. SINGLE SELECT QUERY LIMITATION (CRITICAL)
    - You MUST formulate the entire database search logic into a SINGLE, unified SELECT SQL query.
    - For multi-step analysis, comparisons, sequential filtering, or matching anomalies, you MUST combine these steps into one query using Common Table Expressions (CTEs) or subqueries.
    - You are strictly prohibited from generating multiple SQL queries.

7. LIMIT CLAUSE LOGIC (CRITICAL)
    - If the user query asks for a singular superlative or extreme value (e.g. "most recent", "largest", "highest", "lowest", "busiest", "earliest", "latest"), the query must return exactly one record. You must order the results appropriately and include a `LIMIT 1` clause.
    - If and only if no specific number of records (like "top five") or singular superlative is requested, do not add a hardcoded LIMIT clause. Limit for pagination will be handled by the system.

8. STRING SENTINELS & MISSING VALUES
    - Database tables populated from parsed raw files (like PDFs, CSVs) often contain string representations of missing values (e.g., 'None', 'nan', 'null', or empty spaces ''). Treat these sentinels as logical NULLs and filter them out ONLY for unique identifier/code columns (such as cheque numbers, reference codes, serial numbers) or when counting/aggregating valid entries (e.g., using `column IS NOT NULL AND column NOT IN ('None', 'nan', 'null', '') AND TRIM(column) <> ''`). Do NOT filter them out for general categorical columns (such as transaction types or branch names) unless they interfere with mathematical computations, as they may represent parsed default values expected in raw outputs.

9. DEDUPLICATION BOUNDS (DISTINCT)
    - Use DISTINCT only when the user question explicitly asks for a count of unique/distinct items or requests unique records (e.g., "list the distinct transaction types" or "report all unique account numbers"). Do NOT use DISTINCT for standard transaction listings, chronological logs, or joins unless duplicates would be logically invalid, as it removes valid duplicate transactions (such as identical amounts on the same day).
"""

        prompt = (
            schema_part
            + domain_part
            + context_part
            + input_part
            + history_part
            + rules_part
            + "\nGenerate a highly precise, standard-compliant query that satisfies the user question, avoiding any previous errors, safety violations, or logical auditing flaws. Output must be valid JSON matching the schema."
        )

        logger.debug(f"Generator Prompt:\n{prompt}")
        return client.generate_structured_output(prompt, GeneratedQuery, temperature=0.0)
