-- ============================================================
-- Day 2: Business KPI Dashboard Queries (PostgreSQL)
-- E-Commerce Customer Churn Prediction
-- ============================================================
-- These queries produce metrics a business stakeholder would
-- see on a dashboard. They quantify the COST of churn and
-- help justify retention campaign spend.
-- ============================================================

-- =====================
-- 1. EXECUTIVE SUMMARY: Overall Churn Impact
-- =====================
SELECT 
    COUNT(*) AS total_customers,
    SUM(churn) AS total_churned,
    COUNT(*) - SUM(churn) AS total_retained,
    ROUND(SUM(churn) * 100.0 / COUNT(*), 2) AS churn_rate_pct,
    ROUND(AVG(CASE WHEN churn = 1 THEN o.cashback_amount END), 2) AS avg_cashback_churned,
    ROUND(AVG(CASE WHEN churn = 0 THEN o.cashback_amount END), 2) AS avg_cashback_retained
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id;


-- =====================
-- 2. REVENUE AT RISK: Estimated loss from churners
-- =====================
-- Assume Customer Lifetime Value (CLV) = 15000 INR per customer
-- Revenue at risk = churned_customers * CLV

WITH revenue_risk AS (
    SELECT 
        preferred_order_cat,
        COUNT(*) AS total_customers,
        SUM(churn) AS churned,
        SUM(churn) * 15000 AS revenue_at_risk_inr,
        ROUND(SUM(o.cashback_amount), 2) AS total_cashback_given
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY preferred_order_cat
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
ORDER BY revenue_at_risk_inr DESC;


-- =====================
-- 3. COMPLAINT RESOLUTION ROI
-- =====================
-- Quantify: If we resolved complaints, how many customers could we save?

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
ORDER BY e.complain;


-- =====================
-- 4. HIGH-VALUE CUSTOMER IDENTIFICATION
-- =====================
-- Find customers with high monetary value who are at risk of churning
-- These are the priority targets for retention campaigns

SELECT 
    c.customer_id,
    c.tenure,
    c.preferred_order_cat,
    o.order_count,
    o.cashback_amount,
    o.day_since_last_order,
    e.complain,
    e.satisfaction_score,
    c.churn,
    CASE 
        WHEN o.cashback_amount > 200 AND e.complain = 1 THEN 'CRITICAL - High Value Complainer'
        WHEN o.cashback_amount > 200 AND o.day_since_last_order > 10 THEN 'HIGH - Dormant Big Spender'
        WHEN e.complain = 1 AND e.satisfaction_score <= 2 THEN 'HIGH - Unhappy Complainer'
        WHEN o.day_since_last_order > 15 THEN 'MEDIUM - Going Dormant'
        ELSE 'LOW'
    END AS risk_level
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
LEFT JOIN engagement e ON c.customer_id = e.customer_id
WHERE c.churn = 0
ORDER BY o.cashback_amount DESC NULLS LAST
LIMIT 50;


-- =====================
-- 5. MONTHLY RETENTION CAMPAIGN BUDGET ESTIMATION
-- =====================
-- If we target top 20% at-risk customers with a 500 INR incentive,
-- what is the campaign cost vs potential revenue saved?

WITH at_risk AS (
    SELECT 
        c.customer_id,
        o.cashback_amount,
        e.complain,
        e.satisfaction_score,
        o.day_since_last_order,
        -- Simple risk score: higher = more risky
        (CASE WHEN e.complain = 1 THEN 3 ELSE 0 END +
         CASE WHEN e.satisfaction_score <= 2 THEN 2 ELSE 0 END +
         CASE WHEN o.day_since_last_order > 10 THEN 2 ELSE 0 END +
         CASE WHEN c.tenure < 6 THEN 1 ELSE 0 END) AS risk_score
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    LEFT JOIN engagement e ON c.customer_id = e.customer_id
    WHERE c.churn = 0
),
risk_ranked AS (
    SELECT 
        *,
        PERCENT_RANK() OVER (ORDER BY risk_score DESC) AS risk_percentile
    FROM at_risk
)
SELECT 
    COUNT(*) AS customers_to_target,
    COUNT(*) * 500 AS campaign_cost_inr,
    COUNT(*) * 15000 AS potential_revenue_saved_inr,
    ROUND(COUNT(*) * 15000.0 / NULLIF(COUNT(*) * 500, 0), 1) AS roi_multiplier,
    ROUND(AVG(risk_score), 2) AS avg_risk_score
FROM risk_ranked
WHERE risk_percentile <= 0.20;
