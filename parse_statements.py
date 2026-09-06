import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Ensure logs are visible and well formatted
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bootstrap_main")

# Load environment variables
load_dotenv()

# Import the parsing and storing function
try:
    from utils.statement_parser import parse_and_store_file_transactions
except ImportError as e:
    logger.error("Failed to import parse_and_store_file_transactions from statement_parser.py.")
    logger.error("Please make sure you are running the script from the root workspace directory.")
    sys.exit(1)

def run_init_db(engine):
    """Executes init_db.sql to setup extensions, tables, and indices if table doesn't exist."""
    with engine.connect() as conn:
        # Check if transactions table exists
        check_query = text("SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'transactions')")
        table_exists = conn.execute(check_query).scalar()
        
        if not table_exists:
            logger.info("Table 'transactions' not found. Executing init_db.sql...")
            init_sql_path = "init_db.sql"
            if not os.path.exists(init_sql_path):
                logger.error(f"init_db.sql not found at '{init_sql_path}'. Cannot initialize database.")
                sys.exit(1)
            
            with open(init_sql_path, "r", encoding="utf-8") as f:
                sql_content = f.read()
            
            # Split by semicolon to run individual statements safely (some drivers fail on multi-statement string)
            # We can run it in a single transaction
            with conn.begin():
                # Remove comments and empty lines
                statements = [s.strip() for s in sql_content.split(";") if s.strip()]
                for statement in statements:
                    conn.execute(text(statement))
            logger.info("Database initialized successfully.")
        else:
            logger.info("Table 'transactions' already exists. Skipping database initialization.")

def main():
    # Database connection parameters
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5433")
    db_name = os.getenv("POSTGRES_DB", "bank_statement_qa")
    
    db_url = os.getenv(
        "DATABASE_URL", 
        f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )
    
    logger.info("Initializing database connection...")
    try:
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection established successfully.")
    except Exception as e:
        logger.error(f"Failed to connect to the database at {db_url.replace(db_password, '****')}: {e}")
        logger.error("Please ensure the PostgreSQL container is running.")
        sys.exit(1)
        
    # Run DB Initialization check
    try:
        run_init_db(engine)
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

    workdir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(workdir, "data", "bank-statements")
    
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory '{data_dir}' does not exist. Creating it now...")
        os.makedirs(data_dir, exist_ok=True)
        logger.info(f"Data directory created. Please place your bank statement PDF files in '{data_dir}' and rerun the script.")
        return
        
    # Scan for PDF files
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.info(f"No PDF files found in '{data_dir}'.")
        return
        
    logger.info(f"Found {len(pdf_files)} PDF file(s) in '{data_dir}'. Starting processing...")
    
    for filename in sorted(pdf_files):
        file_path = os.path.join(data_dir, filename)
        logger.info(f"Checking if file '{filename}' has already been parsed...")
        
        # Check if file has already been parsed
        try:
            with Session() as session:
                query = text("SELECT 1 FROM transactions WHERE file_name = :file_name LIMIT 1")
                result = session.execute(query, {"file_name": filename}).first()
                
            if result is not None:
                logger.info(f"Skipping '{filename}': at least one transaction already exists in the database.")
                continue
        except Exception as e:
            logger.error(f"Error querying database for file '{filename}': {e}")
            logger.info("Skipping this file to avoid potential duplicate processing issues.")
            continue
            
        # File has not been parsed yet, prepare output directory
        output_dir = os.path.join(workdir, "output", "parsing", "custom_parser", filename)
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Parsing '{filename}'... Output will be saved to '{output_dir}'")
        
        try:
            with Session() as session:
                parse_and_store_file_transactions(file_path, output_dir, session)
            logger.info(f"Successfully processed and stored transactions for '{filename}'.")
        except Exception as e:
            logger.error(f"Failed to process file '{filename}': {e}")

if __name__ == "__main__":
    main()
