import os
import json
import time
import random
import logging
from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.genai.errors import APIError
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class APIKeyPool:
    """Manages a pool of API keys, handling rotation, dead-key detection, and key selection."""
    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("API key pool cannot be empty")
        self.original_keys = list(keys)
        self.active_keys = list(keys)
        self.key_index = random.randint(0, len(self.active_keys) - 1) if self.active_keys else 0
        # Track last call time per key for precise rate-limiting
        self.last_used_time: Dict[str, float] = {key: 0.0 for key in keys}

    def get_current_key(self) -> str:
        if not self.active_keys:
            raise ValueError("No active API keys remaining in the pool.")
        return self.active_keys[self.key_index]

    def rotate_key(self):
        if not self.active_keys:
            return
        self.key_index = (self.key_index + 1) % len(self.active_keys)

    def mark_dead_key(self):
        if not self.active_keys:
            return
        dead_key = self.active_keys.pop(self.key_index)
        logger.warning(f"Gemini API key {dead_key[:8]}... identified as DEAD/INVALID. Permanently removed from pool.")
        if self.active_keys:
            self.key_index = self.key_index % len(self.active_keys)
        else:
            self.key_index = 0

    def update_last_used(self, key: str):
        self.last_used_time[key] = time.time()

    def get_last_used(self, key: str) -> float:
        return self.last_used_time.get(key, 0.0)

    def active_key_count(self) -> int:
        return len(self.active_keys)


class GeminiLLMClient:
    """Wrapper around Gemini API supporting rotation, rate limiting, and backoff."""
    def __init__(self, model_name: str = "gemini-2.5-flash", min_delay_seconds: float = 2.0):
        self.model_name = model_name
        self.min_delay_seconds = min_delay_seconds
        
        # Load keys from environment
        keys_env = os.getenv("GEMINI_API_KEYS")
        if not keys_env:
            raise ValueError("GEMINI_API_KEYS environment variable is not set or empty.")
        
        try:
            keys = json.loads(keys_env)
            if not isinstance(keys, list) or not keys:
                raise ValueError()
        except Exception:
            # Fallback to splitting by comma if it's not a JSON list
            keys = [k.strip() for k in keys_env.split(",") if k.strip()]
            
        self.key_pool = APIKeyPool(keys)

    def _is_dead_key_error(self, e: Exception) -> bool:
        err_str = str(e).lower()
        if any(msg in err_str for msg in ["api_key_invalid", "api key not valid", "invalid api key", "key not found", "api key is invalid"]):
            return True
        if isinstance(e, APIError):
            if e.code == 400 and "key" in err_str:
                return True
            if e.code == 403:
                return True
        return False

    def _is_rate_limit_error(self, e: Exception) -> bool:
        err_str = str(e).lower()
        if any(msg in err_str for msg in [
            "resource_exhausted", "429", "quota exceeded", "rate limit", "usage limit",
            "503", "unavailable", "high demand", "temp_overloaded", "spikes in demand",
            "504", "deadline_exceeded", "deadline expired"
        ]):
            return True
        if isinstance(e, APIError) and (e.code in (429, 503, 504)):
            return True
        return False

    def generate_structured_output(
        self,
        prompt: str,
        response_schema: Type[BaseModel],
        temperature: float = 0.0,
        max_attempts: int = 15
    ) -> Any:
        """Generates structured JSON output from Gemini with error handling, rotation, and backoff."""
        attempt = 0
        backoff_delay = 1.0

        while attempt < max_attempts:
            if self.key_pool.active_key_count() == 0:
                raise ValueError("All API keys in the pool have been marked as dead. Cannot complete LLM call.")

            key = self.key_pool.get_current_key()
            
            # Enforce local rate limiting on current key
            now = time.time()
            elapsed = now - self.key_pool.get_last_used(key)
            if elapsed < self.min_delay_seconds:
                sleep_time = self.min_delay_seconds - elapsed
                logger.debug(f"Rate limit safety: sleeping {sleep_time:.2f}s for key {key[:8]}...")
                time.sleep(sleep_time)

            logger.info(f"Invoking Gemini with key {key[:8]}... (attempt {attempt+1}/{max_attempts})")
            
            try:
                self.key_pool.update_last_used(key)
                client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=150_000))
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=temperature,
                        seed=42
                    )
                )

                # Return structured object parsed from JSON response
                output_json = json.loads(response.text)
                return response_schema.model_validate(output_json)

            except Exception as e:
                logger.error(f"Gemini API invocation failed with key {key[:8]}...: {e}")

                if self._is_dead_key_error(e):
                    # Permanently remove the dead key
                    self.key_pool.mark_dead_key()
                    # Rotate key immediately to try another one without sleeping
                    continue
                
                elif self._is_rate_limit_error(e):
                    # Rotate key to try other keys first
                    logger.warning("Rate limit / 503 / 504 hit. Rotating key and backing off.")
                    self.key_pool.rotate_key()
                    
                    # Sleep with exponential backoff on current attempt
                    time.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 2.0, 30.0)
                    attempt += 1
                else:
                    # General network or client error: rotate and back off
                    self.key_pool.rotate_key()
                    time.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 1.5, 30.0)
                    attempt += 1

        raise RuntimeError(f"Failed to generate structured output after {max_attempts} attempts across the key pool.")
