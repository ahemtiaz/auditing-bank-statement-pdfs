import os
import math
import json
import fitz
import random
import tempfile
import string
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.parser import parse
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from PIL import Image
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv
from llama_parse import LlamaParse

import time
from functools import wraps
from threading import Lock

from openpyxl.utils import get_column_letter
from openpyxl.styles import Font, PatternFill, Border, Side
from .column import *

load_dotenv()
_LOGGER = logging.getLogger(__name__)


class APIKeyPool:
    def __init__(self, keys):
        assert len(keys) > 0, "API key pool cannot be empty"
        self.keys = keys
        self.key_index = random.randint(0, len(keys) - 1)

    def get_current_key(self):
        return self.keys[self.key_index]

    def rotate_key(self):
        self.key_index = (self.key_index + 1) % len(self.keys)


_GEMINI_API_KEYPOOL = APIKeyPool(json.loads(os.getenv("GEMINI_API_KEYS")))
_LLAMA_CLOUD_API_KEYPOOL = APIKeyPool(json.loads(os.getenv("LLAMA_CLOUD_API_KEYS")))


def resize_image_width(image, new_width=768 * 2):
    width, height = image.size
    if width <= new_width:
        return image

    aspect_ratio = height / width
    new_height = int(aspect_ratio * new_width)
    _LOGGER.info("Resizing image from %s to %s", image.size, (new_width, new_height))

    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image


def rate_limit(min_delay_seconds):
    lock = Lock()
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                now = time.time()
                elapsed = now - last_called[0]
                if elapsed < min_delay_seconds:
                    sleep_time = min_delay_seconds - elapsed
                    time.sleep(sleep_time)

                last_called[0] = time.time()
            return func(*args, **kwargs)

        return wrapper

    return decorator


@retry(
    reraise=True,
    stop=stop_after_attempt(len(_GEMINI_API_KEYPOOL.keys)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    before_sleep=before_sleep_log(_LOGGER, logging.INFO),
)
@rate_limit(min_delay_seconds=6)
def get_gemini_structured_output(
    prompt,
    schema,
    image=None,
    model_name="gemini-2.5-flash",
    temperature=0,
    seed=42,
):
    _GEMINI_CLIENT = genai.Client(api_key=_GEMINI_API_KEYPOOL.get_current_key())
    try:
        response = _GEMINI_CLIENT.models.generate_content(
            model=model_name,
            contents=(
                [resize_image_width(image), prompt] if image else prompt
            ),  # putting image first acc. to gemini docs
            config={
                "thinking_config": types.ThinkingConfig(thinking_budget=0),
                "response_mime_type": "application/json",
                "response_schema": schema,
                "temperature": temperature,
                "seed": seed,
            },
        )

        _LOGGER.info("Token usage data: %s", response.usage_metadata)
        output = json.loads(response.text)

        if not output:
            raise Exception("Gemini usage limit likely exceeded")

        return output
    except Exception as e:
        _LOGGER.info("Rotating Gemini API key")
        _GEMINI_API_KEYPOOL.rotate_key()
        raise e


def convert_page_to_pdf(src_doc, page_no, pdf_path):
    new_doc = fitz.open()
    new_doc.insert_pdf(
        src_doc,
        from_page=page_no,
        to_page=page_no,
        start_at=0,
    )
    new_doc.save(pdf_path)
    new_doc.close()


@retry(
    reraise=True,
    stop=stop_after_attempt(len(_LLAMA_CLOUD_API_KEYPOOL.keys)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    before_sleep=before_sleep_log(_LOGGER, logging.INFO),
)
@rate_limit(min_delay_seconds=6)
def get_llamaparse_output(
    src_doc,
    page_no,
    parsing_instruction=None,
    system_instruction_append=None,
):
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
        pdf_path = os.path.abspath(temp_pdf.name)
        convert_page_to_pdf(src_doc, page_no, pdf_path)

        kwargs = {
            "api_key": _LLAMA_CLOUD_API_KEYPOOL.get_current_key(),
            "target_pages": "0",
            "user_prompt": parsing_instruction,
            "adaptive_long_table": True,
            "skip_diagonal_text": True,
            "system_prompt_append": system_instruction_append,
            "parse_mode": "parse_page_with_agent",
            "model": (
                "anthropic-sonnet-4.0"
                if is_page_scanned(src_doc[page_no])
                else "anthropic-sonnet-3.5"
            ),
        }
        _LOGGER.info("Invoking llamaparse with agentic setting")

        try:
            parser = LlamaParse(**kwargs)
            response = parser.get_json_result(pdf_path)
            if not response:
                raise Exception("llamaparse usage limit likely exceeded")
            return response
        except Exception as e:
            _LOGGER.info("Rotating LlamaParse API key")
            _LLAMA_CLOUD_API_KEYPOOL.rotate_key()
            raise e


def read_text(path):
    with open(path) as f:
        return f.read()


def write_text(text, path):
    with open(path, "w", encoding="utf-8", errors="ignore") as f:
        print(text, file=f)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(data, path):
    with open(path, "w", encoding="utf-8", errors="ignore") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def read_page_image(page, dpi=300):
    pixmap = page.get_pixmap(dpi=dpi)
    return pixmap.pil_image()


def load_or_store_cache(cache_path, func, *args, **kwargs):
    if cache_path.endswith(".json"):
        read_fn, write_fn = read_json, write_json
    else:
        read_fn, write_fn = read_text, write_text

    if os.path.isfile(cache_path):
        try:
            data = read_fn(cache_path)
        except:
            data = {}

        if data:
            return data

    result = func(*args, **kwargs)
    write_fn(result, cache_path)
    return result


def is_page_scanned(page):
    image_covering, text_covering = False, False
    images = page.get_images(full=True)

    if images:
        image_bbox = page.get_image_bbox(images[0])
        page_bbox = page.rect
        intersection_area = image_bbox.intersect(page_bbox).get_area()
        ratio = abs(intersection_area) / abs(page_bbox.get_area())

        if math.isclose(ratio, 1.0):
            image_covering = True

    text_covering = bool(page.get_text().strip())
    return image_covering or not text_covering


def check_convertible(data, conversion_func, replace_comma=False):
    for value in data:
        processed_value = value.strip()

        if replace_comma:
            processed_value = processed_value.replace(",", "")
        try:
            conversion_func(processed_value)
        except ValueError:
            return False

    return True


def check_if_date_and_convert(val):
    if not isinstance(val, str) or not val or check_convertible(val, float, True):
        return False

    default_date = datetime(1000, 1, 1)
    converted_date = None
    try:
        converted_date = parse(val, default=default_date)
    except:
        return False

    if converted_date.year == default_date.year:
        return False
    else:
        return converted_date


def create_formatted_excel(df: pd.DataFrame, output_path: str):
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Transactions")

        workbook = writer.book
        worksheet = writer.sheets["Transactions"]

        header_font = Font(bold=True)
        header_fill = PatternFill(
            start_color="D3D3D3", end_color="D3D3D3", fill_type="solid"
        )
        cell_border = Border(
            left=Side(style="thin"),
            right=Side(style="thin"),
            top=Side(style="thin"),
            bottom=Side(style="thin"),
        )

        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.border = cell_border

        for row in worksheet.iter_rows(min_row=2):
            for cell in row:
                cell.border = cell_border

        for column in worksheet.columns:
            max_length = max(len(str(cell.value)) for cell in column)
            adjusted_width = max_length + 2
            worksheet.column_dimensions[get_column_letter(column[0].column)].width = (
                min(adjusted_width, 30)
            )


def _get_time_tuple(parsed_row):
    """
    Extracts a time tuple from a parsed row for sorting purposes.
    
    Args:
        parsed_row (dict): A dictionary containing parsed column values.
        
    Returns:
        tuple: A tuple containing (year, month, day, hour, minute) for chronological sorting.
        
    If date information is not available, returns (0, 0, 0, 0, 0).
    """
    time = 0
    if parsed_row["Date"] is not None:
        date = parsed_row["Date"]
        if parsed_row["Time"] is not None:
            time = parsed_row["Time"]
            return date.year, date.month, date.day, time.hour, time.minute
        else:
            return date.year, date.month, date.day, 0, 0
    return 0, 0, 0, 0, 0


def get_standardized_df(df):
    """
    Processes an Dataframe and extracts structured data according to column type definitions.
    
    Args:
        df: Pandas Dataframe.
        
    Returns:
        list: A list of dictionaries, where each dictionary contains parsed column values.
        
    The function:
    1. Determines column types
    2. Parses values in each column based on their type
    3. Handles errors during parsing and prints warnings
    """
    df = df.fillna('').astype(str)
    
    column_factories = {}
    for col in df.columns:
        factory = Data.type(col)
        if factory is not None:
            column_factories[col] = factory
        else:
            print(f"Warning: Unknown column type for header: {col}")

    parsed_data = []
    for _, row in df.iterrows():
        parsed_row = {v.name: None for v in INCLUDE_COLUMNS}
        for col, factory in column_factories.items():
            value = row[col].strip()
            if not value or value == '-':
                continue
            try:
                if factory.cls == Union[Debit, Credit]:
                    continue
                parsed = factory.create(value)
                values = parsed.get_columns()
                for v in values:
                    if v.name in parsed_row.keys():
                        parsed_row[v.name] = v

            except ValueError as e:
                print(f"Error parsing value '{value}' in column '{col}' of type {get_type_name(factory.cls)}: {e}")
            except Exception as e:
                print(f"Unexpected error parsing value '{value}' in column '{col}' of type {get_type_name(factory.cls)}: {e}")
        parsed_data.append(parsed_row)

    parsed_data = sorted(parsed_data, key=_get_time_tuple)

    output_df = pd.DataFrame(parsed_data)
    output_df = output_df.dropna(axis=1, how='all')
    return output_df
