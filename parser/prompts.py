import jinja2

COMMON_INFO_SCHEMA = {
    "type": "object",
    "description": "Extracted data directly from a bank statement page image. Contains summary information (like balances, account no, bank name and headers) found on this specific page image.",
    "properties": {
        "opening_balance": {
            "type": "number",
            "format": "double",
            "description": "Analyze the image to find the statement's opening/starting balance. Extract the numeric value, ignoring currency symbols/commas.",
        },
        "account_no": {
            "type": "string",
            "description": "Analyze the image to find the account number for this statement.",
        },
        "bank_name": {
            "type": "string",
            "description": "Analyze the image (including logos and headers) to identify the name of the bank or financial institution.",
        },
        "table_headers": {
            "type": "array",
            "description": "Analyze the image to find the main transaction table. Identify and extract the exact column headers (e.g., ['Date', 'Narration', 'Debit', 'Credit', 'Balance']). ",
            "items": {"type": "string", "description": "A single column header title."},
        },
        "credit_header": {
            "type": "string",
            "description": "The exact column header in the transaction table that represents the credit amount.",
        },
        "debit_header": {
            "type": "string",
            "description": "The exact column header in the transaction table that represents the debit amount.",
        },
        "balance_header": {
            "type": "string",
            "description": "The exact column header in the transaction table that represents the balance amount.",
        },
    },
    "required": [
        "opening_balance",
        "bank_name",
        "account_no",
        "table_headers",
        "credit_header",
        "debit_header",
        "balance_header",
    ],
}

COMMON_INFO_PROMPT = jinja2.Template(
    """You are a financial expert specializing in understanding and parsing bank statements. Analyze the provided bank statement page image. Your goal is to perform OCR on the image and extract the relevant information according to the instructions below, structuring it as a JSON object matching the provided schema.

**Instructions:**

*   Analyze the image to extract statement summary info (e.g., opening balance, account number, bank name)
*   Analyze the image to find the main transaction table. Identify and extract the `table_headers`.
*   Identify the exact column headers for the credit, debit, and balance amounts in the transaction table. These headers may not be explicitly labeled as "Credit", "Debit", or "Balance". Use your financial expertise to determine the correct headers based on the context of the bank statement."""
)

LLAMAPARSE_PARSING_INSTRUCTION = jinja2.Template(
    """You are given a page from a bank statement. The page contains a list of financial transactions within a table, along with other possible text segments outside the table. It is guaranteed that the table has ONLY the following {{num_columns}} columns sequentially from left to right: {{columns_list}}. First, clearly identify which text segments are part of the table. After that, when converting this table to markdown, ensure each transaction contains exactly {{num_columns}} column values, NO MORE OR NO LESS. Extract non-table text segments as usual."""
)

LLAMAPARSE_SYSTEM_APPEND_INSTRUCTION = jinja2.Template(
    """Always prefix any "|" character in the input document with the '\\' character."""
)
