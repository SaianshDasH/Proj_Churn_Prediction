"""
utils.py -- Shared Utility Functions (PostgreSQL Version)
E-Commerce Customer Churn Prediction Project

Reusable helpers for PostgreSQL database operations, display
formatting, and common data science tasks used across all 10 days.
"""

import psycopg2
from psycopg2 import sql
import pandas as pd
import os

# Try to load environment variables from a local .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "ecommerce_churn",
    "user": "postgres",
    "password": "your_password_here",   # <-- Use .env file instead of updating here
}


# ============================================================
# DATABASE UTILITIES
# ============================================================

def get_db_connection(db_config: dict = None) -> psycopg2.extensions.connection:
    """
    Create and return a PostgreSQL database connection.
    
    Args:
        db_config: Dict with host, port, database, user, password.
                   Defaults to DB_CONFIG.
    
    Returns:
        psycopg2 connection object.
    """
    import os
    config = db_config or DB_CONFIG
    
    host = os.environ.get("DB_HOST", config.get("host", "localhost"))
    port = int(os.environ.get("DB_PORT", config.get("port", 5432)))
    database = os.environ.get("DB_NAME", config.get("database", "ecommerce_churn"))
    user = os.environ.get("DB_USER", config.get("user", "postgres"))
    password = os.environ.get("DB_PASSWORD", config.get("password", "postgres"))
    
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    conn.autocommit = True
    return conn


def get_admin_connection(db_config: dict = None) -> psycopg2.extensions.connection:
    """
    Connect to the default 'postgres' database (for creating/dropping databases).
    """
    import os
    config = (db_config or DB_CONFIG).copy()
    config["database"] = "postgres"
    
    host = os.environ.get("DB_HOST", config.get("host", "localhost"))
    port = int(os.environ.get("DB_PORT", config.get("port", 5432)))
    database = os.environ.get("DB_NAME", config.get("database", "postgres"))
    user = os.environ.get("DB_USER", config.get("user", "postgres"))
    password = os.environ.get("DB_PASSWORD", config.get("password", "postgres"))
    
    conn = psycopg2.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    conn.autocommit = True
    return conn


def create_database(db_name: str = "ecommerce_churn", db_config: dict = None):
    """
    Create the project database if it doesn't exist.
    
    Args:
        db_name: Name of the database to create.
        db_config: Connection config dict.
    """
    conn = get_admin_connection(db_config)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
    )
    exists = cursor.fetchone()
    
    if not exists:
        from psycopg2 import sql
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
        print(f"[OK] Created database: {db_name}")
    else:
        print(f"[OK] Database already exists: {db_name}")
    
    cursor.close()
    conn.close()


def run_sql_file(conn, sql_file_path: str) -> None:
    """
    Execute all SQL statements from a .sql file.
    
    Args:
        conn: Active PostgreSQL connection.
        sql_file_path: Path to the SQL file to execute.
    """
    with open(sql_file_path, 'r') as f:
        sql_script = f.read()
    
    cursor = conn.cursor()
    cursor.execute(sql_script)
    conn.commit()
    cursor.close()
    print(f"[OK] Executed: {sql_file_path}")




def run_sql_query(conn, query: str, params: tuple = None) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.
    
    Args:
        conn: Active PostgreSQL connection.
        query: SQL query string.
        params: Optional tuple of query parameters.
    
    Returns:
        pandas DataFrame with query results.
    """
    return pd.read_sql_query(query, conn, params=params)


def run_sql_file_queries(conn, sql_file_path: str) -> list:
    """
    Execute each query from a SQL file individually and return results.
    Splits on semicolons and runs each statement.
    
    Args:
        conn: Active PostgreSQL connection.
        sql_file_path: Path to the SQL file.
    
    Returns:
        List of (query_description, DataFrame) tuples for SELECT statements.
    """
    with open(sql_file_path, 'r') as f:
        sql_content = f.read()
    
    # Split into individual statements
    statements = [s.strip() for s in sql_content.split(';') if s.strip()]
    
    results = []
    for stmt in statements:
        # Skip comments-only blocks
        clean = '\n'.join(
            line for line in stmt.split('\n') 
            if not line.strip().startswith('--')
        ).strip()
        
        if not clean:
            continue
        
        if clean.upper().startswith(('SELECT', 'WITH')):
            try:
                df = pd.read_sql_query(stmt + ';', conn)
                # Extract the comment above as description
                lines = stmt.split('\n')
                desc = next(
                    (l.replace('--', '').strip() for l in lines if l.strip().startswith('--')),
                    'Query Result'
                )
                results.append((desc, df))
            except Exception as e:
                print(f"[!!] Query failed: {str(e)[:80]}")
        else:
            try:
                cursor = conn.cursor()
                cursor.execute(stmt + ';')
                conn.commit()
                cursor.close()
            except Exception as e:
                print(f"[!!] Statement failed: {str(e)[:80]}")
    
    return results


# ============================================================
# DISPLAY UTILITIES
# ============================================================

def print_section_header(title: str, width: int = 60) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_query_result(description: str, df: pd.DataFrame) -> None:
    """Print a query result with a description header."""
    print(f"\n{'-' * 50}")
    print(f">> {description}")
    print(f"{'-' * 50}")
    print(df.to_string(index=False))
    print()


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print comprehensive info about a DataFrame."""
    print_section_header(f"{name} -- Summary")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"\nColumn Types:")
    print(df.dtypes.value_counts().to_string())
    print(f"\nNull Counts:")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 0:
        for col, count in nulls.items():
            pct = count / len(df) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")
    else:
        print("  No null values found!")
    print()


# ============================================================
# FILE & PATH UTILITIES
# ============================================================

def ensure_directories() -> None:
    """Create required project directories if they don't exist."""
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "reports",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[OK] Project directories verified.")


def get_project_root() -> str:
    """Get the project root directory path."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
