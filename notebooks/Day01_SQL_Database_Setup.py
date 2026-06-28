"""
============================================================
Day 01: PostgreSQL Database Setup & Data Ingestion
E-Commerce Customer Churn Prediction
============================================================

OBJECTIVE:
    - Create PostgreSQL database and normalized tables
    - Load raw e-commerce customer data from CSV/Excel
    - Ingest data into the relational schema
    - Run data profiling queries to understand data quality
    - Document initial findings

SKILLS DEMONSTRATED:
    - PostgreSQL: CREATE TABLE, SERIAL, CHECK constraints, Indexes
    - SQL: JOINs, GROUP BY, Aggregations, Window Functions
    - Python: psycopg2, pandas, data ingestion pipeline
    
DATASET:
    Download from Kaggle and place in data/raw/ directory:
    https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction

    Expected file: data/raw/E Commerce Dataset.xlsx
    (or data/raw/ecommerce_churn.csv)
============================================================
"""

import psycopg2
import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import (
    get_db_connection, create_database, run_sql_file, run_sql_query,
    print_section_header, print_query_result, print_dataframe_info,
    ensure_directories, DB_CONFIG
)


# ============================================================
# STEP 1: Setup Project Directories
# ============================================================
print_section_header("STEP 1: Project Setup")
ensure_directories()


# ============================================================
# STEP 2: Load Raw Data
# ============================================================
print_section_header("STEP 2: Load Raw Data")

# Try multiple possible file locations/formats
DATA_PATHS = [
    "data/raw/E Commerce Dataset.xlsx",
    "data/raw/E_Commerce_Dataset.xlsx",
    "data/raw/ecommerce_churn.csv",
    "data/raw/E Commerce.csv",
    "data/raw/ecommerce.csv",
]

raw_df = None
for path in DATA_PATHS:
    if os.path.exists(path):
        print(f"[OK] Found dataset: {path}")
        if path.endswith('.xlsx'):
            try:
                raw_df = pd.read_excel(path, sheet_name='E Comm')
            except Exception:
                raw_df = pd.read_excel(path, sheet_name=0)
        else:
            raw_df = pd.read_csv(path)
        break

if raw_df is None:
    print("[!] Dataset not found in data/raw/ directory.")
    print("    Please download from:")
    print("    https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction")
    print()
    print("    Place the file in: data/raw/")
    print()
    print("[*] Generating synthetic dataset for demonstration...")
    print()
    
    # ========================================
    # Generate synthetic data for demo
    # ========================================
    np.random.seed(42)
    
    n_customers = 5630
    
    raw_df = pd.DataFrame({
        'CustomerID': range(1, n_customers + 1),
        'Churn': np.random.choice([0, 1], size=n_customers, p=[0.83, 0.17]),
        'Tenure': np.random.choice(
            [np.nan] * 100 + list(range(0, 62)), size=n_customers
        ).astype(float),
        'PreferredLoginDevice': np.random.choice(
            ['Mobile Phone', 'Computer', 'Phone'], size=n_customers, p=[0.5, 0.3, 0.2]
        ),
        'CityTier': np.random.choice([1, 2, 3], size=n_customers, p=[0.5, 0.3, 0.2]),
        'WarehouseToHome': np.random.choice(
            [np.nan] * 200 + list(range(5, 130)), size=n_customers
        ).astype(float),
        'PreferredPaymentMode': np.random.choice(
            ['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery', 'E wallet', 'CC'],
            size=n_customers, p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1]
        ),
        'Gender': np.random.choice(['Male', 'Female'], size=n_customers, p=[0.6, 0.4]),
        'HourSpendOnApp': np.random.choice(
            [np.nan] * 100 + list(range(0, 6)), size=n_customers
        ).astype(float),
        'NumberOfDeviceRegistered': np.random.randint(1, 7, size=n_customers),
        'PreferedOrderCat': np.random.choice(
            ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'],
            size=n_customers, p=[0.3, 0.25, 0.2, 0.15, 0.1]
        ),
        'SatisfactionScore': np.random.randint(1, 6, size=n_customers),
        'MaritalStatus': np.random.choice(
            ['Married', 'Single', 'Divorced'], size=n_customers, p=[0.4, 0.35, 0.25]
        ),
        'NumberOfAddress': np.random.randint(1, 23, size=n_customers),
        'Complain': np.random.choice([0, 1], size=n_customers, p=[0.72, 0.28]),
        'OrderAmountHikeFromlable': np.random.choice(
            [np.nan] * 100 + list(range(11, 27)), size=n_customers
        ).astype(float),
        'CouponUsed': np.random.choice(
            [np.nan] * 200 + list(range(0, 17)), size=n_customers
        ).astype(float),
        'OrderCount': np.random.choice(
            [np.nan] * 200 + list(range(1, 17)), size=n_customers
        ).astype(float),
        'DaySinceLastOrder': np.random.choice(
            [np.nan] * 300 + list(range(0, 47)), size=n_customers
        ).astype(float),
        'CashbackAmount': np.round(np.random.uniform(0, 325, size=n_customers), 2),
    })
    
    # Make churners more likely to have complained and low satisfaction
    churn_mask = raw_df['Churn'] == 1
    churn_count = churn_mask.sum()
    
    raw_df.loc[churn_mask, 'Complain'] = np.random.choice(
        [0, 1], size=churn_count, p=[0.4, 0.6]
    )
    raw_df['SatisfactionScore'] = raw_df['SatisfactionScore'].astype(float)
    raw_df.loc[churn_mask, 'SatisfactionScore'] = np.random.choice(
        [1, 2, 3, 4, 5], size=churn_count, p=[0.3, 0.25, 0.2, 0.15, 0.1]
    ).astype(float)
    
    raw_df['Tenure'] = raw_df['Tenure'].astype(float)
    raw_df.loc[churn_mask, 'Tenure'] = np.random.choice(
        list(range(0, 15)) + [np.nan]*20, size=churn_count
    ).astype(float)
    
    print(f"[OK] Generated synthetic dataset with {n_customers} customers")

# Standardize column names (handle variations in Kaggle dataset)
column_mapping = {
    'CustomerID': 'customer_id',
    'Churn': 'churn',
    'Tenure': 'tenure',
    'PreferredLoginDevice': 'preferred_login_device',
    'CityTier': 'city_tier',
    'WarehouseToHome': 'warehouse_to_home',
    'PreferredPaymentMode': 'preferred_payment_mode',
    'Gender': 'gender',
    'HourSpendOnApp': 'hours_on_app',
    'NumberOfDeviceRegistered': 'number_of_device_registered',
    'PreferedOrderCat': 'preferred_order_cat',
    'SatisfactionScore': 'satisfaction_score',
    'MaritalStatus': 'marital_status',
    'NumberOfAddress': 'number_of_address',
    'Complain': 'complain',
    'OrderAmountHikeFromlable': 'order_amount_hike_from_last_year',
    'OrderAmountHikeFromLastYear': 'order_amount_hike_from_last_year',
    'CouponUsed': 'coupon_used',
    'OrderCount': 'order_count',
    'DaySinceLastOrder': 'day_since_last_order',
    'CashbackAmount': 'cashback_amount',
}

raw_df.rename(columns=column_mapping, inplace=True)

# Display raw data info
print_dataframe_info(raw_df, "Raw Dataset")
print("\nFirst 5 rows:")
print(raw_df.head().to_string())


# ============================================================
# STEP 3: Create PostgreSQL Database & Tables
# ============================================================
print_section_header("STEP 3: Create PostgreSQL Database & Tables")

# Create database if it doesn't exist
create_database("ecommerce_churn")

# Connect to the project database
conn = get_db_connection()
print(f"[OK] Connected to PostgreSQL: {DB_CONFIG['database']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}")

# Execute the schema creation SQL
run_sql_file(conn, "sql/01_create_tables.sql")

# Verify tables were created
tables = run_sql_query(conn, """
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public' 
    ORDER BY table_name
""")
print(f"\n[OK] Tables created: {', '.join(tables['table_name'].tolist())}")


# ============================================================
# STEP 4: Ingest Data into Normalized Tables
# ============================================================
print_section_header("STEP 4: Data Ingestion into PostgreSQL")

# Helper function to insert DataFrame into PostgreSQL table
def insert_dataframe(conn, df, table_name, columns):
    """Insert a DataFrame into a PostgreSQL table using executemany."""
    subset = df[columns].copy()
    
    # Replace NaN with None for PostgreSQL
    subset = subset.where(pd.notnull(subset), None)
    
    placeholders = ', '.join(['%s'] * len(columns))
    col_names = ', '.join(columns)
    insert_query = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"
    
    cursor = conn.cursor()
    
    # Convert to list of tuples
    records = [tuple(row) for row in subset.values]
    
    # Use executemany for batch insert
    cursor.executemany(insert_query, records)
    conn.commit()
    cursor.close()
    
    print(f"[OK] Inserted {len(records)} rows into '{table_name}' table")


# ---- Insert into customers table ----
customers_cols = [
    'customer_id', 'gender', 'marital_status', 'city_tier', 'tenure',
    'preferred_login_device', 'preferred_payment_mode', 'preferred_order_cat', 'churn'
]
insert_dataframe(conn, raw_df, 'customers', customers_cols)

# ---- Insert into orders table ----
orders_cols = [
    'customer_id', 'order_count', 'order_amount_hike_from_last_year',
    'coupon_used', 'day_since_last_order', 'cashback_amount'
]
insert_dataframe(conn, raw_df, 'orders', orders_cols)

# ---- Insert into engagement table ----
engagement_cols = [
    'customer_id', 'hours_on_app', 'number_of_address', 'complain',
    'satisfaction_score', 'number_of_device_registered', 'warehouse_to_home'
]
insert_dataframe(conn, raw_df, 'engagement', engagement_cols)


# ============================================================
# STEP 5: Data Integrity Validation
# ============================================================
print_section_header("STEP 5: Data Integrity Validation")

# Check row counts match
for table in ['customers', 'orders', 'engagement']:
    count = run_sql_query(conn, f"SELECT COUNT(*) as cnt FROM {table}")
    print(f"  {table}: {count['cnt'].values[0]} rows")

# Check referential integrity (all customer_ids match)
orphan_check = run_sql_query(conn, """
    SELECT 'orders' AS table_name, COUNT(*) AS orphan_count
    FROM orders o
    WHERE NOT EXISTS (SELECT 1 FROM customers c WHERE c.customer_id = o.customer_id)
    UNION ALL
    SELECT 'engagement', COUNT(*)
    FROM engagement e
    WHERE NOT EXISTS (SELECT 1 FROM customers c WHERE c.customer_id = e.customer_id)
""")
print("\nOrphan records (should be 0):")
print(orphan_check.to_string(index=False))


# ============================================================
# STEP 6: Run Data Profiling Queries
# ============================================================
print_section_header("STEP 6: Data Profiling Results")

# --- 6.1: Churn Distribution ---
print("\n>> 6.1 -- Churn Distribution (Class Balance)")
result = run_sql_query(conn, """
    SELECT 
        churn,
        COUNT(*) AS customer_count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage
    FROM customers
    GROUP BY churn
    ORDER BY churn
""")
print(result.to_string(index=False))

# --- 6.2: Null Value Audit ---
print("\n>> 6.2 -- Null Value Audit")
for table, cols in [
    ('customers', ['gender', 'marital_status', 'city_tier', 'tenure', 
                   'preferred_login_device', 'preferred_payment_mode', 'preferred_order_cat']),
    ('orders', ['order_count', 'order_amount_hike_from_last_year', 'coupon_used',
                'day_since_last_order', 'cashback_amount']),
    ('engagement', ['hours_on_app', 'number_of_address', 'complain',
                    'satisfaction_score', 'number_of_device_registered', 'warehouse_to_home'])
]:
    null_parts = [f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS null_{c}" for c in cols]
    query = f"SELECT '{table}' AS table_name, COUNT(*) AS total, {', '.join(null_parts)} FROM {table}"
    result = run_sql_query(conn, query)
    print(f"\n  Table: {table}")
    for col in cols:
        null_col = f"null_{col}"
        if null_col in result.columns:
            null_count = result[null_col].values[0]
            if null_count > 0:
                pct = null_count / result['total'].values[0] * 100
                print(f"    {col}: {null_count} nulls ({pct:.1f}%)")
    if all(result[f"null_{c}"].values[0] == 0 for c in cols if f"null_{c}" in result.columns):
        print(f"    [OK] No null values!")

# --- 6.3: Descriptive Statistics (with PostgreSQL PERCENTILE_CONT) ---
print("\n>> 6.3 -- Descriptive Statistics (with Median via PERCENTILE_CONT)")
stats_queries = [
    ("tenure", "customers"),
    ("order_count", "orders"),
    ("cashback_amount", "orders"),
    ("day_since_last_order", "orders"),
    ("satisfaction_score", "engagement"),
]
stats_rows = []
for col, table in stats_queries:
    result = run_sql_query(conn, f"""
        SELECT 
            '{col}' AS feature,
            MIN({col}) AS min_val,
            MAX({col}) AS max_val,
            ROUND(AVG({col})::NUMERIC, 2) AS mean_val,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {col}) AS median_val,
            ROUND(STDDEV({col})::NUMERIC, 2) AS std_dev,
            COUNT({col}) AS non_null
        FROM {table}
        WHERE {col} IS NOT NULL
    """)
    stats_rows.append(result)

stats_df = pd.concat(stats_rows, ignore_index=True)
print(stats_df.to_string(index=False))

# --- 6.4: Churn Rate by Key Categories ---
print("\n>> 6.4 -- Churn Rate by Category")

categories = [
    ('Gender', 'gender', 'customers'),
    ('City Tier', 'city_tier', 'customers'),
    ('Payment Mode', 'preferred_payment_mode', 'customers'),
    ('Order Category', 'preferred_order_cat', 'customers'),
    ('Marital Status', 'marital_status', 'customers'),
]

for label, col, table in categories:
    result = run_sql_query(conn, f"""
        SELECT 
            {col},
            COUNT(*) AS total,
            SUM(churn) AS churned,
            ROUND(SUM(churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
        FROM {table}
        GROUP BY {col}
        ORDER BY churn_rate_pct DESC
    """)
    print(f"\n  Churn Rate by {label}:")
    print("  " + result.to_string(index=False).replace('\n', '\n  '))

# --- 6.5: Complain vs Churn ---
print("\n>> 6.5 -- Complain vs Churn (Cross-Tab)")
result = run_sql_query(conn, """
    SELECT 
        e.complain,
        c.churn,
        COUNT(*) AS customer_count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY e.complain), 2) 
            AS pct_within_complain
    FROM customers c
    JOIN engagement e ON c.customer_id = e.customer_id
    GROUP BY e.complain, c.churn
    ORDER BY e.complain, c.churn
""")
print(result.to_string(index=False))


# ============================================================
# STEP 7: Save Processed Data for Day 2+
# ============================================================
print_section_header("STEP 7: Export Processed Data")

# Save the full joined dataset as CSV for easy access later
full_df = run_sql_query(conn, """
    SELECT 
        c.*,
        o.order_count, o.order_amount_hike_from_last_year,
        o.coupon_used, o.day_since_last_order, o.cashback_amount,
        e.hours_on_app, e.number_of_address, e.complain,
        e.satisfaction_score, e.number_of_device_registered, e.warehouse_to_home
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    LEFT JOIN engagement e ON c.customer_id = e.customer_id
""")
full_df.to_csv("data/processed/churn_data_clean.csv", index=False)
print(f"[OK] Saved joined dataset: data/processed/churn_data_clean.csv ({len(full_df)} rows)")

# Close connection
conn.close()
print("\n[OK] PostgreSQL connection closed.")


# ============================================================
# SUMMARY
# ============================================================
print_section_header("DAY 1 -- COMPLETE")
print("""
What we accomplished today:
-----------------------------------------
1. [DONE] Created PostgreSQL database 'ecommerce_churn'
2. [DONE] Designed normalized schema (3 tables with constraints & indexes)
3. [DONE] Ingested 5,630 customers via psycopg2 batch insert
4. [DONE] Validated data integrity (0 orphans, FK checks)
5. [DONE] Ran comprehensive data profiling:
     - Churn distribution (class imbalance check)
     - Null value audit across all tables
     - Descriptive stats with PERCENTILE_CONT (median)
     - Churn rate by 5 categorical dimensions
     - Complain vs Churn cross-tabulation
6. [DONE] Exported processed data for Day 2

PostgreSQL Features Used:
-----------------------------------------
  SERIAL, VARCHAR, NUMERIC, CHECK constraints
  CASCADE drops, Indexes, information_schema
  PERCENTILE_CONT, STDDEV, Window Functions (OVER)
  psycopg2 batch inserts (executemany)

Tomorrow (Day 2):
-----------------------------------------
  -> Advanced SQL: RFM segmentation with CTEs,
     cohort analysis with window functions,
     and business KPI queries
""")
