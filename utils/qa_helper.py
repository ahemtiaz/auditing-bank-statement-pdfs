import os
import re
import json
import logging
import importlib
from typing import List, Dict, Any, Callable
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

from sql_agent.agent import SQLAgent

# Database Connection Setup
db_user = os.getenv("POSTGRES_USER", "postgres")
db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
db_host = os.getenv("POSTGRES_HOST", "localhost")
db_port = os.getenv("POSTGRES_PORT", "5433")
db_name = os.getenv("POSTGRES_DB", "bank_statement_qa")

db_url = os.getenv(
    "DATABASE_URL", 
    f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)

# Initialize SQLAlchemy Engine
engine = create_engine(db_url)

_sql_agent_instance = None
_cached_db_schema = None

def execute_query(query: str) -> List[Dict[str, Any]]:
    """Safely executes a query on the database and returns rows as dictionaries."""
    with engine.connect() as conn:
        result = conn.execute(text(query))
        
        # Check if query returned rows (e.g. SELECT)
        if result.returns_rows:
            # SQLAlchemy 2.0 returns mapping structures which we can convert to dictionaries
            return [dict(row) for row in result.mappings().all()]
        return []

def get_db_schema() -> str:
    """Gets the schema of the transactions table, enriched dynamically with column statistics if the DB is active."""
    global _cached_db_schema
    if _cached_db_schema is not None:
        return _cached_db_schema

    static_schema = """
        CREATE TABLE transactions (
            id UUID PRIMARY KEY,
            file_name VARCHAR(50),
            date DATE,
            trans_type VARCHAR(50),
            cheque VARCHAR(50),
            description TEXT,
            debit NUMERIC(15, 2),
            credit NUMERIC(15, 2),
            bank_name VARCHAR(255),
            page INTEGER,
            account_no VARCHAR(50),
            passed_validation BOOLEAN,
            serial INTEGER,
            time VARCHAR(50),
            value_date DATE,
            reference VARCHAR(50),
            batch_number VARCHAR(50),
            tracer_number VARCHAR(50),
            instrument_number VARCHAR(50),
            trans_code VARCHAR(50),
            branch_code VARCHAR(50),
            branch_name VARCHAR(255),
            search_text TEXT -- indexed with gin_trgm_ops, populated with concatenation of all text and VARCHAR fields for efficient searching
        );
        """

    columns = [
        {"name": "id", "type": "UUID PRIMARY KEY", "cat": "text"},
        {"name": "file_name", "type": "VARCHAR(50)", "cat": "text"},
        {"name": "date", "type": "DATE", "cat": "date"},
        {"name": "trans_type", "type": "VARCHAR(50)", "cat": "text"},
        {"name": "cheque", "type": "VARCHAR(50)", "cat": "text"},
        {"name": "description", "type": "TEXT", "cat": "text"},
        {"name": "debit", "type": "NUMERIC(15, 2)", "cat": "numeric"},
        {"name": "credit", "type": "NUMERIC(15, 2)", "cat": "numeric"},
        {"name": "bank_name", "type": "VARCHAR(255)", "cat": "text"},
        {"name": "page", "type": "INTEGER", "cat": "numeric"},
        {"name": "account_no", "type": "VARCHAR(50)", "cat": "text"},
        {"name": "passed_validation", "type": "BOOLEAN", "cat": "boolean"},
        {"name": "serial", "type": "INTEGER", "cat": "numeric"},
        {"name": "time", "type": "VARCHAR(50)", "cat": "text"},
        {"name": "value_date", "type": "DATE", "cat": "date"},
        {"name": "reference", "type": "VARCHAR(50)", "cat": "text"},
        {"name": "batch_number", "type": "VARCHAR(50)", "cat": "text"},
        {"name": "tracer_number", "type": "VARCHAR(50)", "cat": "text"},
        {"name": "instrument_number", "type": "VARCHAR(50)", "cat": "text"},
        {"name": "trans_code", "type": "VARCHAR(50)", "cat": "text"},
        {"name": "branch_code", "type": "VARCHAR(50)", "cat": "text"},
        {"name": "branch_name", "type": "VARCHAR(255)", "cat": "text"},
        {"name": "search_text", "type": "TEXT", "cat": "text"}
    ]

    try:
        with engine.connect() as conn:
            # Check if database is alive and transactions table exists
            conn.execute(text("SELECT 1 FROM transactions LIMIT 1;"))

            select_parts = []
            for col in columns:
                col_name = col["name"]
                safe_col = f'"{col_name}"'
                select_parts.append(f"COUNT(*) FILTER (WHERE {safe_col} IS NULL) AS {col_name}_nulls")
                select_parts.append(f"COUNT(DISTINCT {safe_col}) AS {col_name}_uniques")
                if col["cat"] in ("numeric", "date"):
                    select_parts.append(f"MIN({safe_col}) AS {col_name}_min")
                    select_parts.append(f"MAX({safe_col}) AS {col_name}_max")

            stats_query = f"SELECT {', '.join(select_parts)} FROM transactions"
            stats_result = conn.execute(text(stats_query)).mappings().first()

            if not stats_result:
                return static_schema

            dynamic_columns_lines = []
            for col in columns:
                col_name = col["name"]
                col_type = col["type"]
                
                nulls_count = stats_result[f"{col_name}_nulls"]
                uniques_count = stats_result[f"{col_name}_uniques"]
                
                comment_parts = [
                    f"Null count: {nulls_count}",
                    f"Unique count: {uniques_count}"
                ]
                
                if uniques_count < 10:
                    if uniques_count == 0:
                        comment_parts.append("Values: (Empty)")
                    else:
                        val_query = f'SELECT DISTINCT "{col_name}" FROM transactions WHERE "{col_name}" IS NOT NULL ORDER BY "{col_name}" LIMIT 10'
                        val_rows = conn.execute(text(val_query)).all()
                        formatted_vals = []
                        for r in val_rows:
                            val = r[0]
                            if val is True:
                                formatted_vals.append("True")
                            elif val is False:
                                formatted_vals.append("False")
                            elif isinstance(val, (int, float)):
                                formatted_vals.append(str(val))
                            else:
                                formatted_vals.append(f"'{str(val)}'")
                        comment_parts.append(f"Values: {', '.join(formatted_vals)}")
                else:
                    if col["cat"] == "numeric":
                        min_val = stats_result[f"{col_name}_min"]
                        max_val = stats_result[f"{col_name}_max"]
                        comment_parts.append(f"Min: {min_val}, Max: {max_val}")
                    elif col["cat"] == "date":
                        min_date = stats_result[f"{col_name}_min"]
                        max_date = stats_result[f"{col_name}_max"]
                        comment_parts.append(f"Earliest: {min_date}, Latest: {max_date}")

                comment_str = ", ".join(comment_parts)
                if col_name == "search_text":
                    comment_str = f"indexed with gin_trgm_ops, populated with concatenation of all text and VARCHAR fields for efficient searching | {comment_str}"
                
                dynamic_columns_lines.append((col_name, col_type, comment_str))

            lines_formatted = []
            for i, (col_name, col_type, comment_str) in enumerate(dynamic_columns_lines):
                if i < len(dynamic_columns_lines) - 1:
                    lines_formatted.append(f"            {col_name} {col_type}, -- {comment_str}")
                else:
                    lines_formatted.append(f"            {col_name} {col_type} -- {comment_str}")

            dynamic_schema = "        CREATE TABLE transactions (\n" + "\n".join(lines_formatted) + "\n        );"
            _cached_db_schema = dynamic_schema
            logger.info("Dynamically generated database schema successfully cached.")
            return dynamic_schema

    except Exception as e:
        logger.error(f"Error generating dynamic database schema: {e}. Falling back to static schema.")
        return static_schema

def initialize_sql_agent() -> SQLAgent:
    """Initializes the SQLAgent orchestrator using a singleton pattern."""
    global _sql_agent_instance
    if _sql_agent_instance is None:
        schema = get_db_schema()
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        domain_context = os.getenv(
            "DATASET_DOMAIN_CONTEXT",
            "The dataset consists of Bangladeshi bank statements. Unless the question states otherwise, "
            "assume Bangladesh conventions throughout (e.g. the weekend is Friday and Saturday; the fiscal "
            "year ends 30 June; the currency is BDT."
        )
        _sql_agent_instance = SQLAgent(
            db_schema=schema,
            executor_fn=execute_query,
            model_name=model_name,
            max_attempts=5,
            min_delay_seconds=1.5, # Cost-efficient and fast rate safety
            domain_context=domain_context
        )
    return _sql_agent_instance

def run_qa_query(query: str) -> Dict[str, Any]:
    """Client utility helper to run the SQL Agent QA flow."""
    agent = initialize_sql_agent()
    return agent.run(query)
