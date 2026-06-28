-- ============================================================
-- Day 1: Data Profiling Queries (PostgreSQL)
-- E-Commerce Customer Churn Prediction
-- ============================================================
-- Run these queries AFTER loading data to understand
-- data quality, distributions, and initial patterns.
-- ============================================================

-- =====================
-- 1. RECORD COUNTS
-- =====================
SELECT 'customers' AS table_name, COUNT(*) AS row_count FROM customers
UNION ALL
SELECT 'orders', COUNT(*) FROM orders
UNION ALL
SELECT 'engagement', COUNT(*) FROM engagement;

-- =====================
-- 2. CHURN DISTRIBUTION
-- =====================
-- Check class imbalance (critical for ML model selection)
SELECT 
    churn,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) AS percentage
FROM customers
GROUP BY churn
ORDER BY churn;

-- =====================
-- 3. NULL VALUE AUDIT
-- =====================
-- Customers table
SELECT 
    'customers' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(*) - COUNT(gender) AS null_gender,
    COUNT(*) - COUNT(marital_status) AS null_marital_status,
    COUNT(*) - COUNT(city_tier) AS null_city_tier,
    COUNT(*) - COUNT(tenure) AS null_tenure,
    COUNT(*) - COUNT(preferred_login_device) AS null_login_device,
    COUNT(*) - COUNT(preferred_payment_mode) AS null_payment_mode,
    COUNT(*) - COUNT(preferred_order_cat) AS null_order_cat
FROM customers;

-- Orders table
SELECT 
    'orders' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(*) - COUNT(order_count) AS null_order_count,
    COUNT(*) - COUNT(order_amount_hike_from_last_year) AS null_order_hike,
    COUNT(*) - COUNT(coupon_used) AS null_coupon_used,
    COUNT(*) - COUNT(day_since_last_order) AS null_days_since_last,
    COUNT(*) - COUNT(cashback_amount) AS null_cashback
FROM orders;

-- Engagement table
SELECT 
    'engagement' AS table_name,
    COUNT(*) AS total_rows,
    COUNT(*) - COUNT(hours_on_app) AS null_hours_app,
    COUNT(*) - COUNT(number_of_address) AS null_addresses,
    COUNT(*) - COUNT(complain) AS null_complain,
    COUNT(*) - COUNT(satisfaction_score) AS null_satisfaction,
    COUNT(*) - COUNT(number_of_device_registered) AS null_devices,
    COUNT(*) - COUNT(warehouse_to_home) AS null_warehouse
FROM engagement;

-- =====================
-- 4. DESCRIPTIVE STATISTICS (PostgreSQL has percentile_cont for median)
-- =====================
SELECT 
    'tenure' AS feature,
    MIN(tenure) AS min_val,
    MAX(tenure) AS max_val,
    ROUND(AVG(tenure), 2) AS mean_val,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tenure) AS median_val,
    ROUND(STDDEV(tenure), 2) AS std_dev,
    COUNT(tenure) AS non_null_count
FROM customers
WHERE tenure IS NOT NULL

UNION ALL

SELECT 
    'order_count',
    MIN(order_count),
    MAX(order_count),
    ROUND(AVG(order_count), 2),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY order_count),
    ROUND(STDDEV(order_count), 2),
    COUNT(order_count)
FROM orders
WHERE order_count IS NOT NULL

UNION ALL

SELECT 
    'cashback_amount',
    MIN(cashback_amount),
    MAX(cashback_amount),
    ROUND(AVG(cashback_amount), 2),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cashback_amount),
    ROUND(STDDEV(cashback_amount)::NUMERIC, 2),
    COUNT(cashback_amount)
FROM orders
WHERE cashback_amount IS NOT NULL

UNION ALL

SELECT 
    'day_since_last_order',
    MIN(day_since_last_order),
    MAX(day_since_last_order),
    ROUND(AVG(day_since_last_order), 2),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY day_since_last_order),
    ROUND(STDDEV(day_since_last_order), 2),
    COUNT(day_since_last_order)
FROM orders
WHERE day_since_last_order IS NOT NULL

UNION ALL

SELECT 
    'hours_on_app',
    MIN(hours_on_app),
    MAX(hours_on_app),
    ROUND(AVG(hours_on_app), 2),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY hours_on_app),
    ROUND(STDDEV(hours_on_app), 2),
    COUNT(hours_on_app)
FROM engagement
WHERE hours_on_app IS NOT NULL

UNION ALL

SELECT 
    'satisfaction_score',
    MIN(satisfaction_score),
    MAX(satisfaction_score),
    ROUND(AVG(satisfaction_score), 2),
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY satisfaction_score),
    ROUND(STDDEV(satisfaction_score)::NUMERIC, 2),
    COUNT(satisfaction_score)
FROM engagement
WHERE satisfaction_score IS NOT NULL;

-- =====================
-- 5. CHURN RATE BY CATEGORICAL FEATURES
-- =====================

-- By Gender
SELECT 
    gender,
    COUNT(*) AS total,
    SUM(churn) AS churned,
    ROUND(SUM(churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY gender
ORDER BY churn_rate_pct DESC;

-- By City Tier
SELECT 
    city_tier,
    COUNT(*) AS total,
    SUM(churn) AS churned,
    ROUND(SUM(churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY city_tier
ORDER BY churn_rate_pct DESC;

-- By Preferred Payment Mode
SELECT 
    preferred_payment_mode,
    COUNT(*) AS total,
    SUM(churn) AS churned,
    ROUND(SUM(churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY preferred_payment_mode
ORDER BY churn_rate_pct DESC;

-- By Preferred Order Category
SELECT 
    preferred_order_cat,
    COUNT(*) AS total,
    SUM(churn) AS churned,
    ROUND(SUM(churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY preferred_order_cat
ORDER BY churn_rate_pct DESC;

-- By Marital Status
SELECT 
    marital_status,
    COUNT(*) AS total,
    SUM(churn) AS churned,
    ROUND(SUM(churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY marital_status
ORDER BY churn_rate_pct DESC;

-- =====================
-- 6. COMPLAIN vs CHURN CROSSTAB
-- =====================
SELECT 
    e.complain,
    c.churn,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(PARTITION BY e.complain), 2) 
        AS pct_within_complain_group
FROM customers c
JOIN engagement e ON c.customer_id = e.customer_id
GROUP BY e.complain, c.churn
ORDER BY e.complain, c.churn;

-- =====================
-- 7. SATISFACTION SCORE vs CHURN
-- =====================
SELECT 
    e.satisfaction_score,
    COUNT(*) AS total_customers,
    SUM(c.churn) AS churned,
    ROUND(SUM(c.churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct
FROM customers c
JOIN engagement e ON c.customer_id = e.customer_id
GROUP BY e.satisfaction_score
ORDER BY e.satisfaction_score;
