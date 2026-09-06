import os
import logging
import pandas as pd
from sqlalchemy import text
from parser.main import process_statement
from utils.data_conversion import (
    parse_date, 
    parse_decimal, 
    parse_integer, 
    parse_boolean, 
    truncate_string
)

logger = logging.getLogger(__name__)

def store_transactions_in_db(df: pd.DataFrame, file_name: str, db_session):
    """
    Store a list of transactions in the database using the provided session.
    """
    insert_count = 0
    error_count = 0
    
    # Convert DataFrame with custom column objects to primitive types
    primitive_df = pd.DataFrame()
    for col in df.columns:
        if df[col].dtype == 'object':
            primitive_df[col] = df[col].apply(
                lambda x: str(x) if hasattr(x, '__str__') and callable(getattr(x, '__str__')) else x
            )
        else:
            primitive_df[col] = df[col]
    
    # Field mapping from DataFrame columns to model attributes
    field_mapping = {
        "date": "Date",
        "trans_type": "Transaction Type",
        "cheque": "Cheque",
        "description": "Description",
        "debit": "Debit",
        "credit": "Credit",
        "bank_name": "Bank Name",
        "page": "Page", 
        "account_no": "Account Number",
        "passed_validation": "Passed Validation",
        "serial": "Serial",
        "time": "Time",
        "value_date": "Value Date",
        "reference": "Reference",
        "batch_number": "Batch Number",
        "tracer_number": "Tracer Number",
        "instrument_number": "Instrument Number",
        "trans_code": "Transaction Code",
        "branch_code": "Branch Code", 
        "branch_name": "Branch Name"
    }

    transactions_data = []

    for _, row in primitive_df.iterrows():
        row_dict = row.to_dict()
        
        # Populate item according to field mapping, using safe gets
        item = {
            "file_name": file_name,
            "date": parse_date(row_dict.get(field_mapping.get("date"))),
            "trans_type": truncate_string(row_dict.get(field_mapping.get("trans_type")), 50),
            "cheque": truncate_string(row_dict.get(field_mapping.get("cheque")), 50),
            "description": row_dict.get(field_mapping.get("description")),
            "debit": parse_decimal(row_dict.get(field_mapping.get("debit"))),
            "credit": parse_decimal(row_dict.get(field_mapping.get("credit"))),
            "bank_name": truncate_string(row_dict.get(field_mapping.get("bank_name")), 255),
            "page": parse_integer(row_dict.get(field_mapping.get("page"))),
            "account_no": truncate_string(row_dict.get(field_mapping.get("account_no")), 50),
            "passed_validation": parse_boolean(row_dict.get(field_mapping.get("passed_validation"))),
            "serial": parse_integer(row_dict.get(field_mapping.get("serial"))),
            "time": truncate_string(row_dict.get(field_mapping.get("time")), 50),
            "value_date": parse_date(row_dict.get(field_mapping.get("value_date"))),
            "reference": truncate_string(row_dict.get(field_mapping.get("reference")), 50),
            "batch_number": truncate_string(row_dict.get(field_mapping.get("batch_number")), 50),
            "tracer_number": truncate_string(row_dict.get(field_mapping.get("tracer_number")), 50),
            "instrument_number": truncate_string(row_dict.get(field_mapping.get("instrument_number")), 50),
            "trans_code": truncate_string(row_dict.get(field_mapping.get("trans_code")), 50),
            "branch_code": truncate_string(row_dict.get(field_mapping.get("branch_code")), 50),
            "branch_name": truncate_string(row_dict.get(field_mapping.get("branch_name")), 255),
        }
        transactions_data.append(item)

    if not transactions_data:
        logger.warning(f"No transactions to store for file {file_name}")
        return

    query = text("""
        INSERT INTO transactions (
            file_name, date, trans_type, cheque, description, debit, credit, 
            bank_name, page, account_no, passed_validation, serial, time, 
            value_date, reference, batch_number, tracer_number, instrument_number, 
            trans_code, branch_code, branch_name
        ) VALUES (
            :file_name, :date, :trans_type, :cheque, :description, :debit, :credit, 
            :bank_name, :page, :account_no, :passed_validation, :serial, :time, 
            :value_date, :reference, :batch_number, :tracer_number, :instrument_number, 
            :trans_code, :branch_code, :branch_name
        )
    """)

    try:
        db_session.execute(query, transactions_data)
        db_session.commit()
        insert_count = len(transactions_data)
        logger.info(f"Successfully inserted {insert_count} transactions for {file_name}")
    except Exception as e:
        db_session.rollback()
        logger.error(f"Failed to insert transactions for {file_name}: {str(e)}")
        raise e


def parse_and_store_file_transactions(file_path: str, output_dir: str, db_session):
    """
    Parse the given file and store the extracted transactions in the database.
    """
    df = process_statement(file_path, output_dir)
    file_name = os.path.basename(file_path)
    store_transactions_in_db(df, file_name, db_session)