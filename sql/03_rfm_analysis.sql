-- ============================================================
-- Day 2: RFM Customer Segmentation (PostgreSQL)
-- E-Commerce Customer Churn Prediction
-- ============================================================
-- RFM = Recency, Frequency, Monetary
-- Industry-standard technique for segmenting customers by
-- purchase behavior. Used by marketing teams to prioritize
-- retention campaigns on high-value at-risk customers.
-- ============================================================

-- =====================
-- STEP 1: Calculate raw RFM values per customer
-- =====================
-- Recency  = day_since_last_order (lower = more recent = better)
-- Frequency = order_count (higher = more orders = better)
-- Monetary  = cashback_amount (proxy for spend; higher = better)

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

-- =====================
-- STEP 2: Assign RFM quintile scores using NTILE
-- =====================
-- NTILE(5) splits sorted data into 5 equal buckets (1-5)
-- For Recency: lower days = better, so we reverse the sort

rfm_scores AS (
    SELECT 
        customer_id,
        churn,
        tenure,
        preferred_order_cat,
        recency,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
    FROM rfm_raw
),

-- =====================
-- STEP 3: Combine scores and assign customer segments
-- =====================

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

-- =====================
-- FINAL OUTPUT: Segment distribution with churn rates
-- =====================
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
ORDER BY churn_rate_pct DESC;
