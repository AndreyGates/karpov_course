SELECT day, length(arrayDistinct(flatten(wau))) as wau FROM 
        (SELECT toDate(timestamp) AS day,
               groupArray(groupArray(DISTINCT user_id))
               OVER (ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as wau
        FROM default.churn_submits
        GROUP BY day) sub