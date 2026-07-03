-- ============================================================
-- Day 2: Cohort Analysis (PostgreSQL)
-- E-Commerce Customer Churn Prediction
-- ============================================================
-- Cohort analysis groups customers by shared characteristics
-- (here: tenure) and tracks churn behavior across groups.
-- This tells us WHEN customers are most likely to leave.
-- ============================================================

-- =====================
-- 1. TENURE-BASED COHORT ANALYSIS
-- =====================
-- Group customers by how long they have been with the company
-- and measure churn risk per cohort

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
ORDER BY churn_rate_pct DESC;


-- =====================
-- 2. PAYMENT MODE COHORT + CHURN RISK
-- =====================
-- Which payment methods are associated with higher churn?
-- Cross-reference with complaint data for deeper insight

SELECT 
    c.preferred_payment_mode,
    e.complain,
    COUNT(*) AS total,
    SUM(c.churn) AS churned,
    ROUND(SUM(c.churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(o.cashback_amount), 2) AS avg_cashback
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN engagement e ON c.customer_id = e.customer_id
GROUP BY c.preferred_payment_mode, e.complain
HAVING COUNT(*) >= 10
ORDER BY churn_rate_pct DESC;


-- =====================
-- 3. ORDER CATEGORY COHORT WITH RUNNING TOTALS
-- =====================
-- Use window functions to show cumulative churn contribution
-- per product category

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
ORDER BY churned DESC;


-- =====================
-- 4. SATISFACTION SCORE COHORT WITH LAG COMPARISON
-- =====================
-- Compare churn rates between adjacent satisfaction levels
-- using the LAG window function

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
    LAG(churn_rate_pct) OVER (ORDER BY satisfaction_score) AS prev_score_churn_rate,
    ROUND(
        churn_rate_pct - COALESCE(LAG(churn_rate_pct) OVER (ORDER BY satisfaction_score), 0), 
        2
    ) AS churn_rate_change
FROM satisfaction_churn
ORDER BY satisfaction_score;


-- =====================
-- 5. CITY TIER x MARITAL STATUS COHORT (PIVOT-STYLE)
-- =====================
-- Cross-tabulation of two demographic dimensions

SELECT 
    city_tier,
    marital_status,
    COUNT(*) AS total,
    SUM(churn) AS churned,
    ROUND(SUM(churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(tenure), 1) AS avg_tenure
FROM customers
GROUP BY city_tier, marital_status
ORDER BY city_tier, churn_rate_pct DESC;
