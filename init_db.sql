CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

CREATE OR REPLACE FUNCTION f_unaccent(text)
  RETURNS text
  LANGUAGE sql IMMUTABLE AS
$func$
SELECT public.unaccent('public.unaccent', $1)
$func$;

CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
    search_text TEXT GENERATED ALWAYS AS (
        lower(
            f_unaccent(
                coalesce(trans_type, '') || ' ' ||
                coalesce(cheque, '') || ' ' ||
                coalesce(description, '') || ' ' ||
                coalesce(bank_name, '') || ' ' ||
                coalesce(reference, '') || ' ' ||
                coalesce(batch_number, '') || ' ' ||
                coalesce(tracer_number, '') || ' ' ||
                coalesce(account_no, '') || ' ' ||
                coalesce(instrument_number, '') || ' ' ||
                coalesce(trans_code, '') || ' ' ||
                coalesce(branch_code, '') || ' ' ||
                coalesce(branch_name, '')
            )
        )
    ) STORED
);

CREATE INDEX IF NOT EXISTS idx_transactions_search_trgm
ON transactions
USING GIN (search_text gin_trgm_ops);
