import time
import re
import logging
from typing import List, Dict, Any, Callable, Optional, Tuple
from pydantic import BaseModel
from .llm_client import GeminiLLMClient
from .query_generator import SQLGenerator, GeneratedQuery
from .query_validator import SQLValidator, ValidationResult

logger = logging.getLogger(__name__)

class SQLAgent:
    """Orchestrates the self-correcting multi-query SQL generation and validation loop with optional validator and RAG context."""
    def __init__(
        self,
        db_schema: str,
        executor_fn: Callable[[str], List[Dict[str, Any]]],
        model_name: str = "gemini-2.5-flash",
        max_attempts: int = 5,
        min_delay_seconds: float = 2.0,
        use_validator: bool = True,
        domain_context: Optional[str] = None
    ):
        self.db_schema = db_schema
        self.executor_fn = executor_fn
        self.model_name = model_name
        self.max_attempts = max_attempts
        self.default_use_validator = use_validator
        
        self.domain_context = domain_context
        
        # Initialize internal classes
        self.llm_client = GeminiLLMClient(model_name=model_name, min_delay_seconds=min_delay_seconds)
        self.generator = SQLGenerator(db_schema)
        self.validator = SQLValidator(db_schema)

    def _extract_count(self, count_rows: List[Any]) -> int:
        """Safely extracts integer count from database query response rows."""
        if not count_rows:
            return 0
        first_row = count_rows[0]
        if isinstance(first_row, dict):
            return int(list(first_row.values())[0])
        elif isinstance(first_row, (list, tuple)):
            return int(first_row[0])
        try:
            return int(first_row)
        except Exception:
            return 0

    def _is_safe_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """Static programmatic analysis to block dangerous updating keywords."""
        # Strip single-line comments (starts with --)
        clean_query = re.sub(r"--.*$", "", query, flags=re.MULTILINE)
        # Strip multi-line comments (/* ... */)
        clean_query = re.sub(r"/\*.*?\*/", "", clean_query, flags=re.DOTALL)

        disallowed = ["update", "delete", "drop", "insert", "alter", "truncate", "create", "grant", "revoke", "replace"]
        # Use regex to find whole words, ignoring case
        for keyword in disallowed:
            pattern = rf"\b{keyword}\b"
            if re.search(pattern, clean_query, re.IGNORECASE):
                return False, f"Query contains dangerous write-operation keyword '{keyword}'."
        return True, None

    def _wrap_query_with_limit(self, query: str, limit: int = 20) -> str:
        """Wraps the query in a limited subquery for safe execution."""
        clean_query = query.strip().rstrip(';')
        return f"SELECT * FROM (\n{clean_query}\n) AS val_subquery LIMIT {limit};"

    def _execute_and_verify_query(
        self,
        query: str
    ) -> Tuple[bool, List[Dict[str, Any]], int, List[str]]:
        """
        Executes and verifies safety for a single SQL query statement.
        Returns:
            Tuple[success_bool, execution_rows, total_count, errors_list]
        """
        # 1. Primary Query Safety Check
        safe, reason = self._is_safe_query(query)
        if not safe:
            err_msg = f"Safety Violation: {reason}"
            logger.warning(err_msg)
            return False, [], 0, [err_msg]

        # 2. Generate and Verify Count Query
        clean_query = query.strip().rstrip(';')
        count_query = f"SELECT COUNT(*) FROM (\n{clean_query}\n) AS count_subquery"
        safe_count, reason_count = self._is_safe_query(count_query)
        if not safe_count:
            err_msg = f"Safety Violation on count query: {reason_count}"
            logger.warning(err_msg)
            return False, [], 0, [err_msg]

        # 3. Execute primary query with a safe limit of 20
        wrapped_query = self._wrap_query_with_limit(query, limit=20)
        logger.info("Executing validation query...")
        try:
            rows = self.executor_fn(wrapped_query)
        except Exception as e:
            err_msg = f"Database Execution Error: {e}"
            logger.error(err_msg)
            return False, [], 0, [err_msg]

        # 4. Execute count query
        logger.info("Executing count query...")
        try:
            count_rows = self.executor_fn(count_query)
            total_count = self._extract_count(count_rows)
        except Exception as e:
            err_msg = f"Database Count Error: {e}"
            logger.error(err_msg)
            return False, [], 0, [err_msg]

        return True, rows, total_count, []

    def run(self, user_query: str, retrieved_context: Optional[str] = None, use_validator: Optional[bool] = None) -> Dict[str, Any]:
        """Runs either the direct single-shot generator or the self-correcting validator loop."""
        start_time = time.time()
        
        domain_context = self.domain_context

        # Decide if validator is needed
        if use_validator is None:
            use_validator = self.default_use_validator

        if not use_validator:
            logger.info("Running standard generator without validator loop (single-shot)...")
            try:
                gen_result: GeneratedQuery = self.generator.generate_query(
                    client=self.llm_client,
                    user_query=user_query,
                    generator_history=[],
                    validator_history=[],
                    retrieved_context=retrieved_context,
                    domain_context=domain_context
                )
                query = gen_result.query
                reasoning = gen_result.reasoning
            except Exception as e:
                err_msg = f"Generation failed: {e}"
                logger.error(err_msg)
                return {
                    "query": "",
                    "results": [],
                    "total_count": 0,
                    "summary": "Generation failed.",
                    "metadata": {
                        "attempts": 1,
                        "execution_time_ms": (time.time() - start_time) * 1000,
                        "validation_passed": False,
                        "generation_error": err_msg
                    }
                }

            # Safety check
            safe, reason = self._is_safe_query(query)
            if not safe:
                err_msg = f"Safety Violation: {reason}"
                logger.error(err_msg)
                return {
                    "query": query,
                    "results": [],
                    "total_count": 0,
                    "summary": "Safety Violation.",
                    "metadata": {
                        "attempts": 1,
                        "execution_time_ms": (time.time() - start_time) * 1000,
                        "validation_passed": False,
                        "error": err_msg
                    }
                }

            # Execute query exactly once (no LIMIT wrap)
            try:
                rows = self.executor_fn(query)
                total_count = len(rows)
                error = None
            except Exception as e:
                rows = []
                total_count = 0
                error = str(e)

            return {
                "query": query,
                "results": rows,
                "total_count": total_count,
                "summary": "Naive single-shot execution. Validation skipped.",
                "reasoning": reasoning,  # Keep reasoning for detailed logging
                "metadata": {
                    "attempts": 1,
                    "execution_time_ms": (time.time() - start_time) * 1000,
                    "validation_passed": False,
                    "had_execution_error": error is not None,
                    "error": error
                }
            }

        # --- Validator/Self-correction Agentic Loop ---
        generator_history: List[Dict[str, Any]] = []
        validator_history: List[Dict[str, Any]] = []
        
        current_query: Optional[str] = None
        execution_rows: List[Dict[str, Any]] = []
        total_count: int = 0
        generator_reasoning: str = ""
        need_generation = True

        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"--- Orchestrator Loop Attempt {attempt}/{self.max_attempts} ---")
            is_last_attempt = (attempt == self.max_attempts)

            if need_generation:
                # 1. Generator Step
                try:
                    gen_result = self.generator.generate_query(
                        self.llm_client, user_query, generator_history, validator_history,
                        retrieved_context, domain_context
                    )
                    generator_reasoning = gen_result.reasoning
                    if not gen_result.query:
                        raise ValueError("Generator returned an empty query.")
                    query = gen_result.query
                except Exception as e:
                    err_msg = f"Generation failed: {e}"
                    logger.error(err_msg)
                    
                    generator_history.append({
                        "query": "",
                        "errors": [err_msg]
                    })
                    need_generation = True
                    continue

                # Execute proposed query
                success, temp_rows, temp_count, attempt_errors = self._execute_and_verify_query(query)
                if not success:
                    generator_history.append({
                        "query": query,
                        "errors": attempt_errors
                    })
                    need_generation = True
                    continue

                # Succeeded executing proposed query
                current_query = query
                execution_rows = temp_rows
                total_count = temp_count
                need_generation = False

            # 2. Validator Step
            try:
                val_result: ValidationResult = self.validator.validate(
                    client=self.llm_client,
                    user_query=user_query,
                    query=current_query,
                    execution_rows=execution_rows,
                    total_count=total_count,
                    is_final=is_last_attempt,
                    generator_history=generator_history,
                    validator_history=validator_history,
                    generator_reasoning=generator_reasoning,
                    retrieved_context=retrieved_context,
                    domain_context=domain_context
                )
            except Exception as e:
                err_msg = f"Validation system failure: {e}"
                logger.error(err_msg)
                validator_history.append({
                    "query": current_query,
                    "issue": err_msg
                })
                need_generation = True
                continue

            if val_result.approved:
                logger.info("Validator APPROVED the generated query!")
                final_rows = self._fetch_client_result(current_query)
                
                if not final_rows and total_count > 0:
                    err_msg = "Final client result fetch failed due to execution error on the larger dataset."
                    logger.error(err_msg)
                    generator_history.append({
                        "query": current_query,
                        "errors": [err_msg]
                    })
                    need_generation = True
                    continue
                else:
                    return {
                        "query": current_query,
                        "results": final_rows,
                        "total_count": total_count,
                        "summary": val_result.analysis_summary,
                        "metadata": {
                            "attempts": attempt,
                            "execution_time_ms": (time.time() - start_time) * 1000,
                            "validation_passed": True,
                        }
                    }

            # If rejected, it's corrected
            logger.info("Validator rejected the current SQL. Attempting verification of correction...")
            issue_msg = val_result.error_message or "Query logic did not fully answer user query."

            # Programmatically use corrected query if provided
            if val_result.corrected_query:
                corrected_query = val_result.corrected_query
            else:
                corrected_query = current_query

            # Record incorrected queries (the rejected queries) and issue in validator history
            validator_history.append({
                "query": current_query,
                "issue": issue_msg
            })

            # Try executing the corrected query
            success, temp_rows, temp_count, corr_errors = self._execute_and_verify_query(corrected_query)
            if not success:
                # Correction execution failed: Record execution error in generator history for next attempt
                generator_history.append({
                    "query": corrected_query,
                    "errors": corr_errors
                })
                need_generation = True
                continue

            # Success executing validator's corrected query
            current_query = corrected_query
            execution_rows = temp_rows
            total_count = temp_count
            need_generation = False

            # If corrected on final attempt, stop and return best effort
            if is_last_attempt:
                logger.warning("Validator corrected query on the final attempt. Returning best-effort results.")
                final_rows = self._fetch_client_result(current_query)
                return {
                    "query": current_query,
                    "results": final_rows,
                    "total_count": total_count,
                    "summary": val_result.analysis_summary or "Failed to answer the question within max attempts.",
                    "metadata": {
                        "attempts": attempt,
                        "execution_time_ms": (time.time() - start_time) * 1000,
                        "validation_passed": False,
                    }
                }

        # Compilation and Fallbacks (Max Attempts Exceeded)
        logger.warning("Agent exhausted maximum attempts. Executing fallback.")
        return {
            "query": "",
            "results": [],
            "total_count": 0,
            "summary": "Failed to answer the question within max attempts.",
            "metadata": {
                "attempts": self.max_attempts,
                "execution_time_ms": (time.time() - start_time) * 1000,
                "validation_passed": False,
            }
        }

    def _fetch_client_result(self, query: str) -> List[Dict[str, Any]]:
        """Fetches all rows for the client-facing output for the approved query."""
        try:
            return self.executor_fn(query)
        except Exception as e:
            logger.error(f"Error fetching final client results: {e}")
            return []
