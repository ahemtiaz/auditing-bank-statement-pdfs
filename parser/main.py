import os
import re
import json
import fitz
import math
import shutil
import string
import logging
import pandas as pd
from datetime import datetime
from dateutil.parser import parse

from .utils import (
    read_page_image,
    is_page_scanned,
    check_convertible,
    load_or_store_cache,
    get_llamaparse_output,
    create_formatted_excel,
    get_gemini_structured_output,
    check_if_date_and_convert,
    get_standardized_df
)
from .prompts import (
    COMMON_INFO_SCHEMA,
    COMMON_INFO_PROMPT,
    LLAMAPARSE_PARSING_INSTRUCTION,
    LLAMAPARSE_SYSTEM_APPEND_INSTRUCTION,
)

_LOGGER = logging.getLogger(__name__)


def check_if_transaction_row_is_valid(row, metadata):
    # check column count match
    if len(row) != len(metadata["table_headers"]):
        _LOGGER.info("Discarding row for column count mismatch: %s", row)
        return False

    # check if at least one column has valid characters
    target_chars = string.ascii_letters + string.digits
    if all(not any(c in target_chars for c in col.lower()) for col in row):
        _LOGGER.info("Discarding row for all invalid columns: %s", row)
        return False

    header2val = dict(zip(metadata["table_headers"], row))
    numeric_cols = ["credit_header", "debit_header", "balance_header"]

    # check if all numeric columns are invalid
    if any(
        not (
            not header2val[metadata[col]]
            or check_convertible([header2val[metadata[col]]], float, True)
        )
        for col in numeric_cols
    ):
        _LOGGER.info("Discarding row for invalid numeric columns: %s", row)
        return False

    def _get_numeric_val(string_val):
        if not string_val:
            return 0.0

        string_val = string_val.replace(",", "")
        return float(string_val)

    # check if row has both credit and debit
    if (
        _get_numeric_val(header2val[metadata["credit_header"]]) != 0.0
        and _get_numeric_val(header2val[metadata["debit_header"]]) != 0.0
    ):
        _LOGGER.info("Discarding row with both credit and debit: %s", row)
        return False

    # check if row dowsn't have credit or debit but has balance
    if (
        not (
            header2val[metadata["credit_header"]]
            or header2val[metadata["debit_header"]]
        )
        and header2val[metadata["balance_header"]]
    ):

        _LOGGER.info(
            "Discarding row with no credit and debit, but has balance: %s", row
        )
        return False

    # check if row has credit and debit but no balance
    if (
        header2val[metadata["credit_header"]]
        and header2val[metadata["debit_header"]]
        and not header2val[metadata["balance_header"]]
    ):

        _LOGGER.info(
            "Discarding row with both credit and debit, but no balance: %s", row
        )
        return False

    # allow empty credit, debit and balance for now (continuation rows)
    return True


def check_if_any_column_is_date(row):
    return any(check_if_date_and_convert(v) for v in row)


def _parse_table(rows, metadata):
    parsed_rows = []
    header_found = False

    def _preprocess_val(val):
        val = val.strip() if val is not None else ""
        list_of_strings = ["<br>", "<br/>", "<br />"]
        for s in list_of_strings:
            val = val.replace(s, "\n")

        if all(c in ",." + string.digits for c in val) and val.count(".") > 1:
            parts = val.split(".")
            val = "".join(parts[:-1]) + "." + parts[-1]

        return val

    for row in rows:
        if any(v is None for v in row):
            _LOGGER.info("Discarding row for None val in column: %s", row)
            continue

        row = [_preprocess_val(v) for v in row]
        # if the page has markdown separators, llamaparse struggles
        # to maintain the column boundaries correctly
        if row and len(row) != len(metadata["table_headers"]):
            col_idx = 0
            while col_idx < len(row) - 1:
                if row[col_idx].endswith("\\"):
                    row[col_idx] = row[col_idx].rstrip("\\") + row[col_idx + 1]
                    del row[col_idx + 1]
                else:
                    col_idx += 1

        # sometimes llamaparse adds empty columns to the right
        if row and len(row) != len(metadata["table_headers"]):
            if not row[-1]:
                while len(row) > len(metadata["table_headers"]):
                    if not row[-1]:
                        del row[-1]
                        continue

                    break

        # sometimes some transaction balance columns have
        # suffix like "CR" attached e.g. midland bank
        if row and len(row) == len(metadata["table_headers"]):
            numeric_cols = ["balance_header"]
            for col in numeric_cols:
                col_idx = metadata["table_headers"].index(metadata[col])
                col_val = row[col_idx]
                converted_val = "".join(
                    k for k in col_val if k in ",. +-" + string.digits
                )

                if not check_convertible([col_val], float, True) and check_convertible(
                    [converted_val], float, True
                ):
                    _LOGGER.info(
                        "Converting %s column value: %s -> %s",
                        col,
                        col_val,
                        converted_val,
                    )
                    row[col_idx] = converted_val

        if row == metadata["table_headers"]:
            if not header_found:
                header_found = True
                parsed_rows = []

        if check_if_transaction_row_is_valid(row, metadata):
            parsed_rows.append(row)

    return parsed_rows, header_found


def parse_tables(tablewise_rows, metadata):
    parsed_rows = []
    header_rows = []

    for table_rows in tablewise_rows:
        parsed_table_rows, header_found = _parse_table(table_rows, metadata)
        header_rows.append(header_found)
        if parsed_table_rows:
            parsed_rows.extend(parsed_table_rows)

    return parsed_rows, any(header_rows)


def parse_tables_in_page_with_naive_grids(page, metadata):
    if is_page_scanned(page):
        return (None, None)

    tables = page.find_tables(strategy="lines_strict")
    tablewise_rows = []

    for tab in tables:
        df = tab.to_pandas()

        if len(df.columns) != len(metadata["table_headers"]):
            continue

        rows = [df.columns.tolist()] + df.values.tolist()
        tablewise_rows.append(rows)

    return parse_tables(tablewise_rows, metadata)


def parse_tables_in_page_with_llamaparse(src_doc, page_no, metadata, page_cache_path):
    instruction = LLAMAPARSE_PARSING_INSTRUCTION.render(
        num_columns=len(metadata["table_headers"]),
        columns_list=metadata["table_headers"],
    )
    response = load_or_store_cache(
        page_cache_path,
        get_llamaparse_output,
        src_doc,
        page_no,
        instruction,
        LLAMAPARSE_SYSTEM_APPEND_INSTRUCTION.render(),
    )
    tablewise_rows = []
    page_result = response[0]["pages"][0]

    for page_item in page_result["items"]:
        if page_item["type"] == "table":
            tablewise_rows.append(page_item["rows"])

    return parse_tables(tablewise_rows, metadata)


def fix_dataframe_dtypes(df, extracted_data):
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            non_empty_strings = [x for x in df[col] if x.strip() != ""]
            can_be_float = check_convertible(
                non_empty_strings, float, replace_comma=True
            )
            can_be_int = check_convertible(non_empty_strings, int)
            # for scanned pages, sometimes ocr gets confused between commas and decimal points
            # e.g. 1,000.00 gets converted to 1,000,00
            if col in [
                extracted_data[v]
                for v in ["credit_header", "debit_header", "balance_header"]
            ]:
                offsets = set()
                for val in non_empty_strings:
                    if "." in val:
                        offset = len(val) - val.rfind(".") - 1
                        offsets.add(offset)

                if len(offsets) == 1:
                    target_offset = offsets.pop()

                    def find_and_fix_numerics(val):
                        target_val = val
                        if "." not in val and "," in val:
                            comma_index = val.rfind(",")

                            if len(val) - comma_index - 1 == target_offset:
                                target_val = (
                                    val[:comma_index] + "." + val[comma_index + 1 :]
                                )

                        return target_val

                    df[col] = df[col].apply(lambda x: find_and_fix_numerics(x))

            if (
                can_be_float
                and non_empty_strings
                and all("." in x for x in non_empty_strings)
            ) or col in [
                extracted_data[v]
                for v in ["credit_header", "debit_header", "balance_header"]
            ]:
                _LOGGER.info("Converting column %s to float", col)
                df[col] = df[col].apply(
                    lambda x: (
                        0.0 if x.strip() == "" else (float(x.strip().replace(",", "")))
                    )
                )

            # if all non-zero debit values are negative, convert to positive. Ex: Agrani
            if col == extracted_data["debit_header"]:
                non_zero_debit = df[col][df[col] != 0]
                if all(v < 0 for v in non_zero_debit) and len(non_zero_debit) > 1:
                    _LOGGER.info("Converting debit column %s to positive", col)
                    df[col] = df[col].apply(lambda x: (abs(x) if x < 0 else x))

    return df


def prepare_dataframe(extracted_data):
    transactions = [
        k
        | {
            "bank_name": extracted_data["bank_name"],
            "stmt_account_no": extracted_data["account_no"],
        }
        for k in extracted_data["transactions"]
    ]

    df = pd.DataFrame(transactions)
    df = fix_dataframe_dtypes(df, extracted_data)

    return df


def validate_transactions(df, extracted_results):
    opening_balance = extracted_results["opening_balance"]
    previous_balance = opening_balance

    balance_col = extracted_results["balance_header"]
    credit_col = extracted_results["credit_header"]
    debit_col = extracted_results["debit_header"]
    meta_columns = ["page_index", "passed_validation"]

    indices_to_remove = []
    # iterate over transactions row by row
    for index, row in df.iterrows():
        transaction_credit = row.get(credit_col, 0)
        transaction_debit = row.get(debit_col, 0)
        current_balance = row.get(balance_col, 0)

        # remove transaction row from df if both credit and debit is 0
        if math.isclose(transaction_credit, 0.0) and math.isclose(
            transaction_debit, 0.0
        ):
            _LOGGER.info(
                "Transaction %d doesn't contain either credit or debit. Details: %s",
                index,
                json.dumps(row.to_dict(), indent=4),
            )
            indices_to_remove.append(index)
            continue
        elif transaction_credit > 0.0 and transaction_debit > 0.0:
            _LOGGER.info(
                "Transaction %d contains both credit and debit. Details: %s",
                index,
                json.dumps(row.to_dict(), indent=4),
            )
            indices_to_remove.append(index)
            continue

        expected_balance = previous_balance + transaction_credit - transaction_debit
        if not math.isclose(current_balance, expected_balance):
            _LOGGER.info(
                "Transaction %d: Expected balance %.2f but got %.2f, attempting reconciliation."
                " Full transaction: %s",
                index,
                expected_balance,
                current_balance,
                json.dumps(row.to_dict(), indent=4),
            )

            # try inverting negative signs (ocr sometimes misreads) Ex: DBBL
            reconciled = False
            if any(
                [
                    v < 0
                    for v in [transaction_credit, transaction_debit, current_balance]
                ]
            ):
                for i in range(1, 8):
                    temp_credit = transaction_credit * (
                        1 if i & 1 or transaction_credit > 0.0 else -1
                    )
                    temp_debit = transaction_debit * (
                        1 if i & 2 or transaction_debit > 0.0 else -1
                    )
                    temp_balance = current_balance * (
                        1 if i & 4 or current_balance > 0.0 else -1
                    )
                    temp_expected_balance = previous_balance + temp_credit - temp_debit

                    if math.isclose(temp_balance, temp_expected_balance):
                        _LOGGER.info(
                            "Transaction %s: Reconciled balance by inverting signs. "
                            "Credit: %.2f -> %.2f, Debit: %.2f -> %.2f, Balance: %.2f -> %.2f",
                            index,
                            transaction_credit,
                            temp_credit,
                            transaction_debit,
                            temp_debit,
                            current_balance,
                            temp_balance,
                        )
                        df.at[index, credit_col] = temp_credit
                        df.at[index, debit_col] = temp_debit
                        df.at[index, balance_col] = temp_balance
                        reconciled = True
                        break

            # try decimal variations to account for OCR errors
            if not reconciled:

                def _get_decimal_variations(val):
                    if val == 0:
                        return [val]

                    variations = [val]
                    sign = 1 if val > 0 else -1
                    val = abs(val)
                    str_digits = str(val).replace(".", "")

                    for i in range(1, len(str_digits)):
                        integer_part = str_digits[:-i]
                        decimal_part = str_digits[-i:]
                        if not integer_part:
                            continue

                        candidate = integer_part + "." + decimal_part
                        try:
                            new_val = float(candidate)
                            variations.append(sign * new_val)
                        except ValueError:
                            continue

                    return sorted(set(variations))

                for t_credit in _get_decimal_variations(transaction_credit):
                    for t_debit in _get_decimal_variations(transaction_debit):
                        for t_balance in _get_decimal_variations(current_balance):

                            reconciled_expected_balance = (
                                previous_balance + t_credit - t_debit
                            )

                            if math.isclose(t_balance, reconciled_expected_balance):
                                _LOGGER.info(
                                    "Transaction %s: Reconciled balance by decimal variations. "
                                    "Credit: %.2f -> %.2f, Debit: %.2f -> %.2f, Balance: %.2f -> %.2f",
                                    index,
                                    transaction_credit,
                                    t_credit,
                                    transaction_debit,
                                    t_debit,
                                    current_balance,
                                    t_balance,
                                )
                                df.at[index, credit_col] = t_credit
                                df.at[index, debit_col] = t_debit
                                df.at[index, balance_col] = t_balance
                                reconciled = True
                                break

                        if reconciled:
                            break

                    if reconciled:
                        break

            # next attempt: swap credit and debit
            if not reconciled:
                reconciled_expected_balance = (
                    previous_balance + transaction_debit - transaction_credit
                )

                if math.isclose(current_balance, reconciled_expected_balance):
                    _LOGGER.info(
                        "Transaction %s: Swapping Credit and Debit reconciled the balance. Full transaction: {%s}",
                        index,
                        json.dumps(row.to_dict(), indent=4),
                    )
                    df.at[index, credit_col] = transaction_debit
                    df.at[index, debit_col] = transaction_credit
                    reconciled = True

            # next attempt: check if opening balance
            if not reconciled:
                if index == 0 and math.isclose(
                    current_balance, extracted_results["opening_balance"]
                ):
                    _LOGGER.info(
                        "Discarding transaction %d with opening balance", index
                    )
                    indices_to_remove.append(index)
                    reconciled = True

            # next attempt: check if any column can be coverted to date
            if not reconciled:
                if not check_if_any_column_is_date(row):
                    _LOGGER.info("Discarding transaction %d without any date", index)
                    indices_to_remove.append(index)
                    reconciled = True

            # we only look for WARNINGS as sign of error
            if not reconciled:
                _LOGGER.warning("Unable to reconcile Transaction %d", index)
                df.at[index, "passed_validation"] = False

        previous_balance = current_balance

    if indices_to_remove:
        df.drop(indices_to_remove, inplace=True)
        _LOGGER.info("Some rows were removed. Rerunning dataframe dtype conversions")
        df = fix_dataframe_dtypes(df, extracted_results)

    if all(k in df.columns for k in meta_columns):
        meta_df = df[meta_columns]
        df.drop(columns=meta_columns, inplace=True)
    else:
        meta_df = pd.DataFrame(columns=meta_columns)

    return df, meta_df


def post_processing(df):
    """Post-processing on the standardized dataframe.

    Runs after get_standardized_df(), so column names are the standardized
    names defined in column.py (e.g. 'Date', 'Debit') and cell values are
    Data objects (with a .value attribute) or None.
    """
    initial_len = len(df)

    # 1. Drop rows where Date is null
    if "Date" in df.columns:
        null_date_mask = df["Date"].isna()
        if null_date_mask.any():
            _LOGGER.info(
                "Post-processing: dropping %d rows with null Date",
                null_date_mask.sum(),
            )
            df = df[~null_date_mask].reset_index(drop=True)

    # 2. Flip negative debit values to positive
    if "Debit" in df.columns:
        for idx in df.index:
            cell = df.at[idx, "Debit"]
            if cell is not None and hasattr(cell, "value") and cell.value < 0:
                _LOGGER.info(
                    "Post-processing: flipping negative debit %.2f at row %d",
                    cell.value, idx,
                )
                cell.value = abs(cell.value)

    if len(df) != initial_len:
        _LOGGER.info(
            "Post-processing: %d rows removed (%d -> %d)",
            initial_len - len(df), initial_len, len(df),
        )

    return df


def process_statement(pdf_path, output_dir):
    basename = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    log_file_path = os.path.join(output_dir, f"{basename}.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _LOGGER.addHandler(file_handler)
    shutil.copy(pdf_path, os.path.join(output_dir, f"{basename}.pdf"))

    src_doc = fitz.open(pdf_path)
    extracted_results = {
        # additional keys will be populated from metadata
        "transactions": [],
    }

    pagewise_transactions = []
    for page_index, page in enumerate(src_doc):
        page_image = read_page_image(page)

        if page_index == 0:
            statement_metadata_path = os.path.join(
                output_dir, f"page_{page_index}.metadata.json"
            )
            statement_metadata = load_or_store_cache(
                statement_metadata_path,
                get_gemini_structured_output,
                COMMON_INFO_PROMPT.render(),
                COMMON_INFO_SCHEMA,
                page_image,
            )
            _LOGGER.info("Got info response for page %d", page_index)
            extracted_results |= statement_metadata

        parsed_rows, header_found = parse_tables_in_page_with_naive_grids(
            page, extracted_results
        )
        if not parsed_rows and not header_found:
            page_cache_path = os.path.join(
                output_dir, f"page_{page_index}.llamaparse.json"
            )
            parsed_rows, header_found = parse_tables_in_page_with_llamaparse(
                src_doc, page_index, extracted_results, page_cache_path
            )
            _LOGGER.info("Got llamaparse response for page %d", page_index)
        else:
            _LOGGER.info("Got naive grid response for page %d", page_index)

        if parsed_rows:
            # handle multi-page rows
            header2val = dict(zip(extracted_results["table_headers"], parsed_rows[0]))
            if (
                pagewise_transactions
                and len(pagewise_transactions[-1]) > 0
                and not (
                    header2val[extracted_results["credit_header"]]
                    or header2val[extracted_results["debit_header"]]
                    or header2val[extracted_results["balance_header"]]
                )
            ):
                _LOGGER.info(
                    "Multi page transaction detected. Last page row: %s, Current page row: %s",
                    pagewise_transactions[-1][-1],
                    parsed_rows[0],
                )

                last_row = pagewise_transactions[-1][-1]
                for col_index, (old, new) in enumerate(zip(last_row, parsed_rows[0])):
                    old = "" if old is None else old
                    new = "" if new is None else new
                    pagewise_transactions[-1][-1][col_index] = (
                        (old.strip() + f" {new.strip()}") if new else old
                    )

                parsed_rows = parsed_rows[1:]

            pagewise_transactions.append(parsed_rows)

    for page_index, page_transactions in enumerate(pagewise_transactions):
        for transaction in page_transactions:
            extracted_results["transactions"].append(
                dict(zip(extracted_results["table_headers"], transaction))
                | {"page_index": page_index + 1, "passed_validation": True}
            )

    df = prepare_dataframe(extracted_results)
    df, meta_df = validate_transactions(df, extracted_results)

    merged_df = pd.concat([df, meta_df], axis=1)
    merged_df = get_standardized_df(merged_df)
    merged_df = post_processing(merged_df)

    output_excel_path = os.path.join(output_dir, f"{basename}.xlsx")

    create_formatted_excel(merged_df, output_excel_path)

    # Clean up resources
    src_doc.close()

    _LOGGER.removeHandler(file_handler)

    return merged_df
