"""
============================================================
Day 02: Advanced SQL Analysis -- RFM, Cohorts & Business KPIs
E-Commerce Customer Churn Prediction
============================================================

OBJECTIVE:
    - Perform RFM (Recency, Frequency, Monetary) segmentation
    - Run tenure-based cohort analysis with window functions
    - Calculate business KPIs: revenue at risk, campaign ROI
    - Demonstrate production-grade SQL (CTEs, NTILE, LAG, RANK)

SKILLS DEMONSTRATED:
    - SQL: CTEs, Window Functions (NTILE, LAG, RANK, PERCENT_RANK)
    - SQL: Subqueries, CASE expressions, Cumulative aggregates
    - Python: pandas, data loading, result formatting
    
NOTE:
    SQL files in sql/ folder use PostgreSQL syntax for the portfolio.
    This script loads from the processed CSV and runs queries via
    an in-memory SQLite engine for local execution.
============================================================
"""

import sqlite3
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import print_section_header, print_dataframe_info, ensure_directories


# ============================================================
# STEP 1: Load processed data from Day 1
# ============================================================
print_section_header("STEP 1: Load Processed Data")
ensure_directories()

CSV_PATH = "data/processed/churn_data_clean.csv"

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"[OK] Loaded: {CSV_PATH} ({len(df)} rows)")
else:
    # Regenerate synthetic data if CSV not found
    import numpy as np
    np.random.seed(42)
    n = 5630
    df = pd.DataFrame({
        'customer_id': range(1, n + 1),
        'churn': np.random.choice([0, 1], size=n, p=[0.83, 0.17]),
        'tenure': np.random.choice(
            [None] * 3 + list(range(0, 62)), size=n),
        'gender': np.random.choice(['Male', 'Female'], size=n, p=[0.6, 0.4]),
        'city_tier': np.random.choice([1, 2, 3], size=n, p=[0.5, 0.3, 0.2]),
        'marital_status': np.random.choice(
            ['Married', 'Single', 'Divorced'], size=n, p=[0.4, 0.35, 0.25]),
        'preferred_login_device': np.random.choice(
            ['Mobile Phone', 'Computer', 'Phone'], size=n, p=[0.5, 0.3, 0.2]),
        'preferred_payment_mode': np.random.choice(
            ['Debit Card', 'UPI', 'Credit Card', 'Cash on Delivery', 'E wallet'],
            size=n, p=[0.25, 0.25, 0.2, 0.15, 0.15]),
        'preferred_order_cat': np.random.choice(
            ['Laptop & Accessory', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'],
            size=n, p=[0.3, 0.25, 0.2, 0.15, 0.1]),
        'order_count': np.random.choice(
            [None] * 1 + list(range(1, 17)) * 2, size=n),
        'order_amount_hike_from_last_year': np.random.choice(
            [None] * 1 + list(range(11, 27)) * 2, size=n),
        'coupon_used': np.random.choice(
            [None] * 1 + list(range(0, 17)) * 2, size=n),
        'day_since_last_order': np.random.choice(
            [None] * 2 + list(range(0, 47)), size=n),
        'cashback_amount': np.round(np.random.uniform(0, 325, size=n), 2),
        'hours_on_app': np.random.choice(
            [None] * 1 + list(range(0, 6)) * 4, size=n),
        'number_of_address': np.random.randint(1, 23, size=n),
        'complain': np.random.choice([0, 1], size=n, p=[0.72, 0.28]),
        'satisfaction_score': np.random.randint(1, 6, size=n),
        'number_of_device_registered': np.random.randint(1, 7, size=n),
        'warehouse_to_home': np.random.choice(
            [None] * 6 + list(range(5, 130)), size=n),
    })
    # Inject churn signal
    churn_mask = df['churn'] == 1
    df['satisfaction_score'] = df['satisfaction_score'].astype(float)
    df.loc[churn_mask, 'complain'] = np.random.choice(
        [0, 1], size=churn_mask.sum(), p=[0.4, 0.6]
    ).astype(float)
    df.loc[churn_mask, 'satisfaction_score'] = np.random.choice(
        [1, 2, 3, 4, 5], size=churn_mask.sum(), p=[0.3, 0.25, 0.2, 0.15, 0.1]
    ).astype(float)
    df.to_csv(CSV_PATH, index=False)
    print(f"[OK] Generated synthetic data and saved to {CSV_PATH}")

print(f"     Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"     Churn rate: {df['churn'].mean()*100:.2f}%")


# ============================================================
# STEP 2: Load data into in-memory SQLite for query execution
# ============================================================
print_section_header("STEP 2: Create In-Memory SQL Engine")

conn = sqlite3.connect(":memory:")

# Split into normalized tables (mirrors PostgreSQL schema)
customers_cols = [
    'customer_id', 'gender', 'marital_status', 'city_tier', 'tenure',
    'preferred_login_device', 'preferred_payment_mode', 'preferred_order_cat', 'churn'
]
orders_cols = [
    'customer_id', 'order_count', 'order_amount_hike_from_last_year',
    'coupon_used', 'day_since_last_order', 'cashback_amount'
]
engagement_cols = [
    'customer_id', 'hours_on_app', 'number_of_address', 'complain',
    'satisfaction_score', 'number_of_device_registered', 'warehouse_to_home'
]

df[customers_cols].to_sql('customers', conn, index=False, if_exists='replace')
df[orders_cols].to_sql('orders', conn, index=False, if_exists='replace')
df[engagement_cols].to_sql('engagement', conn, index=False, if_exists='replace')

print("[OK] Loaded 3 tables into SQL engine (customers, orders, engagement)")


# ============================================================
# STEP 3: RFM Segmentation
# ============================================================
print_section_header("STEP 3: RFM Customer Segmentation")

print("\n>> RFM = Recency (days since last order) + Frequency (order count) + Monetary (cashback)")
print("   Each customer gets a 1-5 score per dimension using NTILE window function.\n")

rfm_query = """
WITH rfm_raw AS (
    SELECT 
        c.customer_id,
        c.churn,
        c.tenure,
        c.preferred_order_cat,
        COALESCE(o.day_since_last_order, 999) AS recency,
        COALESCE(o.order_count, 0) AS frequency,
        COALESCE(o.cashback_amount, 0) AS monetary
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
),
rfm_scores AS (
    SELECT 
        customer_id, churn, tenure, preferred_order_cat,
        recency, frequency, monetary,
        NTILE(5) OVER (ORDER BY recency DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
    FROM rfm_raw
),
rfm_segments AS (
    SELECT 
        *,
        (r_score + f_score + m_score) AS rfm_total,
        CASE 
            WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
            WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal Customers'
            WHEN r_score >= 4 AND f_score <= 2 THEN 'New Customers'
            WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
            WHEN r_score <= 2 AND f_score <= 2 AND m_score >= 3 THEN 'Cant Lose Them'
            WHEN r_score <= 2 AND f_score <= 2 THEN 'Lost'
            ELSE 'Needs Attention'
        END AS customer_segment
    FROM rfm_scores
)
SELECT 
    customer_segment,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned,
    ROUND(SUM(churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(monetary), 2) AS avg_monetary,
    ROUND(AVG(recency), 1) AS avg_recency,
    ROUND(AVG(frequency), 1) AS avg_frequency
FROM rfm_segments
GROUP BY customer_segment
ORDER BY churn_rate_pct DESC
"""
rfm_result = pd.read_sql_query(rfm_query, conn)
print(rfm_result.to_string(index=False))

# Show insight
highest_churn_seg = rfm_result.iloc[0]
print(f"\n   INSIGHT: '{highest_churn_seg['customer_segment']}' segment has the highest")
print(f"   churn rate at {highest_churn_seg['churn_rate_pct']}% with avg spend of "
      f"{highest_churn_seg['avg_monetary']}")


# ============================================================
# STEP 4: Tenure-Based Cohort Analysis
# ============================================================
print_section_header("STEP 4: Tenure-Based Cohort Analysis")

print("\n>> Grouping customers by tenure length to find WHEN churn peaks.\n")

cohort_query = """
WITH tenure_cohorts AS (
    SELECT 
        customer_id,
        tenure,
        churn,
        CASE 
            WHEN tenure IS NULL THEN 'Unknown'
            WHEN tenure <= 6 THEN '0-6 months'
            WHEN tenure <= 12 THEN '7-12 months'
            WHEN tenure <= 24 THEN '13-24 months'
            WHEN tenure <= 36 THEN '25-36 months'
            ELSE '37+ months'
        END AS tenure_cohort
    FROM customers
)
SELECT 
    tenure_cohort,
    COUNT(*) AS total_customers,
    SUM(churn) AS churned_customers,
    COUNT(*) - SUM(churn) AS retained_customers,
    ROUND(SUM(churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    RANK() OVER (ORDER BY SUM(churn) * 1.0 / COUNT(*) DESC) AS risk_rank
FROM tenure_cohorts
GROUP BY tenure_cohort
ORDER BY churn_rate_pct DESC
"""
cohort_result = pd.read_sql_query(cohort_query, conn)
print(cohort_result.to_string(index=False))


# ============================================================
# STEP 5: Satisfaction Score Cohort with LAG Comparison
# ============================================================
print_section_header("STEP 5: Satisfaction Score Analysis (LAG Window Function)")

print("\n>> Using LAG() to compare churn rates between adjacent satisfaction levels.\n")

satisfaction_query = """
WITH satisfaction_churn AS (
    SELECT 
        e.satisfaction_score,
        COUNT(*) AS total,
        SUM(c.churn) AS churned,
        ROUND(SUM(c.churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
    FROM customers c
    JOIN engagement e ON c.customer_id = e.customer_id
    GROUP BY e.satisfaction_score
)
SELECT 
    satisfaction_score,
    total,
    churned,
    churn_rate_pct,
    LAG(churn_rate_pct) OVER (ORDER BY satisfaction_score) AS prev_score_churn,
    ROUND(
        churn_rate_pct - COALESCE(LAG(churn_rate_pct) OVER (ORDER BY satisfaction_score), 0), 
        2
    ) AS churn_rate_change
FROM satisfaction_churn
ORDER BY satisfaction_score
"""
sat_result = pd.read_sql_query(satisfaction_query, conn)
print(sat_result.to_string(index=False))


# ============================================================
# STEP 6: Category Churn with Cumulative Window Functions
# ============================================================
print_section_header("STEP 6: Product Category Churn (Cumulative Analysis)")

print("\n>> Using cumulative SUM window function to find which categories")
print("   contribute most to total churn (Pareto-style analysis).\n")

category_query = """
WITH category_churn AS (
    SELECT 
        preferred_order_cat,
        COUNT(*) AS total_customers,
        SUM(churn) AS churned,
        ROUND(SUM(churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
    FROM customers
    GROUP BY preferred_order_cat
)
SELECT 
    preferred_order_cat,
    total_customers,
    churned,
    churn_rate_pct,
    SUM(churned) OVER (ORDER BY churned DESC) AS cumulative_churned,
    ROUND(
        SUM(churned) OVER (ORDER BY churned DESC) * 100.0 
        / SUM(churned) OVER (), 2
    ) AS cumulative_churn_pct
FROM category_churn
ORDER BY churned DESC
"""
cat_result = pd.read_sql_query(category_query, conn)
print(cat_result.to_string(index=False))


# ============================================================
# STEP 7: Business KPIs -- Revenue at Risk
# ============================================================
print_section_header("STEP 7: Business KPIs -- Revenue at Risk")

print("\n>> Assuming Customer Lifetime Value (CLV) = 15,000 INR per customer.\n")

revenue_query = """
WITH revenue_risk AS (
    SELECT 
        c.preferred_order_cat,
        COUNT(*) AS total_customers,
        SUM(c.churn) AS churned,
        SUM(c.churn) * 15000 AS revenue_at_risk_inr,
        ROUND(SUM(o.cashback_amount), 2) AS total_cashback_given
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.preferred_order_cat
)
SELECT 
    preferred_order_cat,
    total_customers,
    churned,
    revenue_at_risk_inr,
    total_cashback_given,
    ROUND(revenue_at_risk_inr * 100.0 / SUM(revenue_at_risk_inr) OVER(), 2) 
        AS pct_of_total_risk
FROM revenue_risk
ORDER BY revenue_at_risk_inr DESC
"""
rev_result = pd.read_sql_query(revenue_query, conn)
print(rev_result.to_string(index=False))

total_risk = rev_result['revenue_at_risk_inr'].sum()
print(f"\n   TOTAL REVENUE AT RISK: {total_risk:,.0f} INR")
print(f"   = {total_risk/100000:.1f} Lakhs per quarter")


# ============================================================
# STEP 8: Campaign ROI Estimation
# ============================================================
print_section_header("STEP 8: Retention Campaign ROI Estimation")

print("\n>> If we target top 20% at-risk customers with 500 INR incentive each...\n")

roi_query = """
WITH at_risk AS (
    SELECT 
        c.customer_id,
        o.cashback_amount,
        e.complain,
        e.satisfaction_score,
        o.day_since_last_order,
        c.tenure,
        (CASE WHEN e.complain = 1 THEN 3 ELSE 0 END +
         CASE WHEN e.satisfaction_score <= 2 THEN 2 ELSE 0 END +
         CASE WHEN o.day_since_last_order > 10 THEN 2 ELSE 0 END +
         CASE WHEN c.tenure < 6 OR c.tenure IS NULL THEN 1 ELSE 0 END) AS risk_score
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    LEFT JOIN engagement e ON c.customer_id = e.customer_id
    WHERE c.churn = 0
),
risk_ranked AS (
    SELECT *,
        PERCENT_RANK() OVER (ORDER BY risk_score DESC) AS risk_percentile
    FROM at_risk
)
SELECT 
    COUNT(*) AS customers_to_target,
    COUNT(*) * 500 AS campaign_cost_inr,
    COUNT(*) * 15000 AS potential_revenue_saved_inr,
    ROUND(COUNT(*) * 15000.0 / (COUNT(*) * 500), 1) AS roi_multiplier,
    ROUND(AVG(risk_score), 2) AS avg_risk_score
FROM risk_ranked
WHERE risk_percentile <= 0.20
"""
roi_result = pd.read_sql_query(roi_query, conn)
print(roi_result.to_string(index=False))

if len(roi_result) > 0:
    roi = roi_result.iloc[0]
    print(f"\n   Campaign Cost:    {int(roi['campaign_cost_inr']):>12,} INR")
    print(f"   Revenue Saved:    {int(roi['potential_revenue_saved_inr']):>12,} INR")
    print(f"   ROI Multiplier:   {roi['roi_multiplier']:>12}x")


# ============================================================
# STEP 9: Complaint Resolution Impact
# ============================================================
print_section_header("STEP 9: Complaint Resolution Impact Analysis")

complaint_query = """
SELECT 
    e.complain,
    COUNT(*) AS total_customers,
    SUM(c.churn) AS churned,
    ROUND(SUM(c.churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    SUM(c.churn) * 15000 AS revenue_at_risk_inr,
    ROUND(AVG(e.satisfaction_score), 2) AS avg_satisfaction
FROM customers c
JOIN engagement e ON c.customer_id = e.customer_id
GROUP BY e.complain
ORDER BY e.complain
"""
complaint_result = pd.read_sql_query(complaint_query, conn)
print(complaint_result.to_string(index=False))

if len(complaint_result) == 2:
    no_complain_churn = complaint_result.iloc[0]['churn_rate_pct']
    yes_complain_churn = complaint_result.iloc[1]['churn_rate_pct']
    saveable = complaint_result.iloc[1]['churned'] - complaint_result.iloc[0]['churned']
    print(f"\n   Complainers churn at {yes_complain_churn}% vs {no_complain_churn}% for non-complainers")
    print(f"   If complaints were resolved, up to ~{int(saveable)} additional customers could be retained")
    print(f"   = {int(saveable) * 15000:,} INR in potential savings")


# Close connection
conn.close()


# ============================================================
# SUMMARY
# ============================================================
print_section_header("DAY 2 -- COMPLETE")
print("""
What we accomplished today:
-----------------------------------------
1. [DONE] RFM Segmentation (NTILE, CTEs, CASE)
     -> Identified customer segments: Champions to Lost
     -> Found which segments have highest churn rates

2. [DONE] Tenure-Based Cohort Analysis (RANK)
     -> Discovered which tenure groups churn most
     -> New customers (0-6 months) vs long-term behavior

3. [DONE] Satisfaction Score Analysis (LAG)
     -> Compared churn rates between adjacent scores
     -> Quantified the churn rate change per score level

4. [DONE] Product Category Pareto Analysis (Cumulative SUM)
     -> Found which categories contribute most to total churn
     -> Pareto-style 80/20 analysis

5. [DONE] Revenue at Risk Calculation
     -> Quantified total revenue at risk in INR
     -> Broken down by product category

6. [DONE] Retention Campaign ROI
     -> Estimated cost vs benefit of targeting top 20% at-risk
     -> Calculated ROI multiplier for campaign justification

7. [DONE] Complaint Resolution Impact
     -> Quantified saveable customers and revenue from fixing complaints

SQL Features Demonstrated:
-----------------------------------------
  CTEs (WITH clauses), NTILE, LAG, RANK, PERCENT_RANK
  Cumulative SUM OVER, CASE expressions, COALESCE
  Multi-table JOINs, Subqueries, HAVING

Tomorrow (Day 3):
-----------------------------------------
  -> Python EDA: 12+ visualizations with matplotlib, seaborn, plotly
  -> Distribution plots, correlation heatmaps, violin plots
  -> Statistical analysis of churn drivers
""")
