SELECT
    toStartOfMonth(toDate(buy_date)) AS month,
    AVG(check_amount) AS avg_check,
    quantilesExactExclusive(0.5)(check_amount)[1] AS median_check
    
FROM view_checks 
GROUP BY month;