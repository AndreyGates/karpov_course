SELECT toDate(timestamp) AS day,
       COUNT(DISTINCT user_id) AS dau
FROM churn_submits
GROUP BY day;
