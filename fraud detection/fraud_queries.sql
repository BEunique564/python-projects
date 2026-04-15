-- ================================================================
-- Fraud Detection — SQL Analytics Queries
-- Author: Vaibhav Gupta
-- Database: transactions (PostgreSQL / MySQL compatible)
-- ================================================================

-- ── Schema ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id      BIGINT PRIMARY KEY,
    user_id             INT,
    amount              DECIMAL(12, 2),
    merchant_id         INT,
    merchant_category   VARCHAR(50),
    transaction_time    TIMESTAMP,
    is_international    BOOLEAN,
    card_age_days       INT,
    distance_from_home  DECIMAL(10, 2),
    is_fraud            BOOLEAN DEFAULT FALSE,
    fraud_score         DECIMAL(5, 4)
);

-- ── 1. Fraud Rate by Hour of Day ─────────────────────────────────
SELECT
    EXTRACT(HOUR FROM transaction_time)  AS hour_of_day,
    COUNT(*)                             AS total_txns,
    SUM(is_fraud::INT)                   AS fraud_count,
    ROUND(AVG(is_fraud::INT) * 100, 2)  AS fraud_rate_pct
FROM transactions
GROUP BY 1
ORDER BY 4 DESC;

-- ── 2. High-Risk Velocity: Users with >5 txns in 1 hour ──────────
WITH ranked AS (
    SELECT
        user_id,
        transaction_time,
        COUNT(*) OVER (
            PARTITION BY user_id
            ORDER BY transaction_time
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
        ) AS txns_last_1h
    FROM transactions
)
SELECT DISTINCT user_id, MAX(txns_last_1h) AS max_velocity
FROM ranked
WHERE txns_last_1h > 5
GROUP BY user_id
ORDER BY 2 DESC
LIMIT 50;

-- ── 3. Merchant Category Fraud Heatmap ───────────────────────────
SELECT
    merchant_category,
    COUNT(*)                                AS total,
    SUM(is_fraud::INT)                      AS frauds,
    ROUND(AVG(is_fraud::INT) * 100, 2)     AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                   AS avg_amount,
    ROUND(AVG(CASE WHEN is_fraud THEN amount END), 2) AS avg_fraud_amount
FROM transactions
GROUP BY merchant_category
HAVING COUNT(*) > 100
ORDER BY fraud_rate_pct DESC;

-- ── 4. International + Night Combo (High-Risk Flag) ──────────────
SELECT
    is_international,
    CASE WHEN EXTRACT(HOUR FROM transaction_time) BETWEEN 0 AND 5
         THEN 'Night' ELSE 'Day' END                   AS time_bucket,
    COUNT(*)                                             AS txn_count,
    ROUND(AVG(is_fraud::INT) * 100, 2)                  AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                                AS avg_amount
FROM transactions
GROUP BY 1, 2
ORDER BY fraud_rate_pct DESC;

-- ── 5. Rolling 7-Day Fraud Rate (Trend) ──────────────────────────
SELECT
    DATE_TRUNC('day', transaction_time)                           AS day,
    COUNT(*)                                                       AS total,
    SUM(is_fraud::INT)                                             AS frauds,
    ROUND(
        AVG(SUM(is_fraud::INT)) OVER (
            ORDER BY DATE_TRUNC('day', transaction_time)
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 2
    )                                                              AS rolling_7d_fraud_avg
FROM transactions
GROUP BY 1
ORDER BY 1;

-- ── 6. New Cards (< 30 days) Fraud Stats ─────────────────────────
SELECT
    CASE WHEN card_age_days < 30  THEN 'New Card (< 30 days)'
         WHEN card_age_days < 180 THEN 'Recent (30–180 days)'
         ELSE 'Established (180+ days)' END         AS card_segment,
    COUNT(*)                                         AS txn_count,
    ROUND(AVG(is_fraud::INT) * 100, 2)              AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                            AS avg_amount
FROM transactions
GROUP BY 1
ORDER BY fraud_rate_pct DESC;

-- ── 7. Top 20 Users by Fraud Loss Exposure ────────────────────────
SELECT
    user_id,
    COUNT(*)                                             AS total_txns,
    SUM(CASE WHEN is_fraud THEN amount ELSE 0 END)      AS total_fraud_amount,
    ROUND(AVG(is_fraud::INT) * 100, 2)                  AS fraud_rate_pct,
    MAX(fraud_score)                                     AS max_risk_score
FROM transactions
GROUP BY user_id
HAVING SUM(CASE WHEN is_fraud THEN amount ELSE 0 END) > 0
ORDER BY total_fraud_amount DESC
LIMIT 20;

-- ── 8. Distance-Bucketed Fraud Rate ───────────────────────────────
SELECT
    CASE
        WHEN distance_from_home < 10   THEN '0–10 km'
        WHEN distance_from_home < 50   THEN '10–50 km'
        WHEN distance_from_home < 200  THEN '50–200 km'
        ELSE '200+ km'
    END                                                  AS distance_bucket,
    COUNT(*)                                             AS total,
    ROUND(AVG(is_fraud::INT) * 100, 2)                  AS fraud_rate_pct
FROM transactions
GROUP BY 1
ORDER BY fraud_rate_pct DESC;