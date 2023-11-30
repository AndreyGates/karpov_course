SELECT DISTINCT sku, dates,
       AVG(price) OVER (PARTITION BY sku, dates) AS price,
       COUNT(*) OVER(PARTITION BY sku, dates) AS qty
FROM transactions;
