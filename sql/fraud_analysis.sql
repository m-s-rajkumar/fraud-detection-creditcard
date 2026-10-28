-- name: 1_overall_fraud_rate
SELECT COUNT(*) AS transactions,
       SUM(Class) AS frauds,
       ROUND(100.0 * SUM(Class) / COUNT(*), 3) AS fraud_rate_pct,
       ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2) AS fraud_loss
FROM transactions;

-- name: 2_fraud_by_hour
SELECT CAST(Time / 3600 AS INTEGER) % 24 AS hour_of_day,
       COUNT(*) AS transactions,
       SUM(Class) AS frauds,
       ROUND(100.0 * SUM(Class) / COUNT(*), 3) AS fraud_rate_pct
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;

-- name: 3_fraud_by_amount_band
SELECT CASE
         WHEN Amount < 10 THEN '1: under 10'
         WHEN Amount < 50 THEN '2: 10-50'
         WHEN Amount < 100 THEN '3: 50-100'
         WHEN Amount < 250 THEN '4: 100-250'
         WHEN Amount < 500 THEN '5: 250-500'
         WHEN Amount < 1000 THEN '6: 500-1000'
         ELSE '7: 1000 and above'
       END AS amount_band,
       COUNT(*) AS transactions,
       SUM(Class) AS frauds,
       ROUND(100.0 * SUM(Class) / COUNT(*), 3) AS fraud_rate_pct,
       ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2) AS fraud_loss
FROM transactions
GROUP BY amount_band
ORDER BY amount_band;

-- name: 4_largest_fraud_transactions
SELECT txn_id, Time, Amount
FROM transactions
WHERE Class = 1
ORDER BY Amount DESC
LIMIT 10;

-- name: 5_cumulative_fraud_loss_by_hour
WITH hourly AS (
  SELECT CAST(Time / 3600 AS INTEGER) AS hour_index,
         SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END) AS fraud_loss
  FROM transactions
  GROUP BY hour_index
)
SELECT hour_index,
       ROUND(fraud_loss, 2) AS fraud_loss,
       ROUND(SUM(fraud_loss) OVER (ORDER BY hour_index), 2) AS cumulative_fraud_loss
FROM hourly
ORDER BY hour_index;

-- name: 6_hours_with_fraud_rate_above_average
WITH hourly AS (
  SELECT CAST(Time / 3600 AS INTEGER) % 24 AS hour_of_day,
         100.0 * SUM(Class) / COUNT(*) AS fraud_rate_pct
  FROM transactions
  GROUP BY hour_of_day
),
overall AS (
  SELECT 100.0 * SUM(Class) / COUNT(*) AS overall_rate FROM transactions
)
SELECT h.hour_of_day,
       ROUND(h.fraud_rate_pct, 3) AS fraud_rate_pct,
       ROUND(h.fraud_rate_pct / o.overall_rate, 2) AS times_the_average
FROM hourly h CROSS JOIN overall o
WHERE h.fraud_rate_pct > o.overall_rate
ORDER BY times_the_average DESC;