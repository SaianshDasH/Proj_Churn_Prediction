"""
utils.py — Shared Utility Functions
E-Commerce Customer Churn Prediction Project

Reusable helpers for database operations, display formatting,
and common data science tasks used across all 10 days.
"""

import sqlite3
import pandas as pd
import os


# ============================================================
# DATABASE UTILITIES
# ============================================================

def get_db_connection(db_path: str = "data/ecommerce_churn.db") -> sqlite3.Connection:
    """
    Create and return a SQLite database connection.
    
    Args:
        db_path: Path to the SQLite database file.
    
    Returns:
        sqlite3.Connection object.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column-name access
    return conn


def run_sql_file(conn: sqlite3.Connection, sql_file_path: str) -> None:
    """
    Execute all SQL statements from a .sql file.
    
    Args:
        conn: Active SQLite connection.
        sql_file_path: Path to the SQL file to execute.
    """
    with open(sql_file_path, 'r') as f:
        sql_script = f.read()
    
    cursor = conn.cursor()
    cursor.executescript(sql_script)
    conn.commit()
    print(f"[OK] Executed: {sql_file_path}")


def run_sql_query(conn: sqlite3.Connection, query: str, params: tuple = None) -> pd.DataFrame:
    """
    Execute a SQL query and return results as a DataFrame.
    
    Args:
        conn: Active SQLite connection.
        query: SQL query string.
        params: Optional tuple of query parameters.
    
    Returns:
        pandas DataFrame with query results.
    """
    if params:
        return pd.read_sql_query(query, conn, params=params)
    return pd.read_sql_query(query, conn)


def run_sql_file_queries(conn: sqlite3.Connection, sql_file_path: str) -> list:
    """
    Execute each query from a SQL file individually and return results.
    Splits on semicolons and runs each statement.
    
    Args:
        conn: Active SQLite connection.
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
        # Skip comments-only blocks and non-SELECT statements
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
                conn.execute(stmt + ';')
                conn.commit()
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
    print_section_header(f"{name} — Summary")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
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
    # Navigate up from src/ to project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
