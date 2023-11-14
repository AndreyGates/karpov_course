/* SELECT
  brand,
  COUNT(sku_type) AS count_sku
FROM
  sku_dict_another_one
GROUP BY
  brand 
ORDER BY
  count_sku DESC,
  brand DESC 
LIMIT
  10
OFFSET
  1; */

/* SELECT
  sku_type,
  COUNT(DISTINCT vendor) AS count_vendor
FROM
  sku_dict_another_one
GROUP BY
  sku_type
ORDER BY
  count_vendor DESC,
  sku_type DESC
LIMIT
  10; */

/* SELECT
  sku_type,
  COUNT(DISTINCT vendor) AS count_vendor
FROM
  sku_dict_another_one
GROUP BY
  sku_type
ORDER BY
  count_vendor DESC,
  sku_type DESC
LIMIT
  10; */

/* SELECT
  vendor,
  COUNT(DISTINCT brand) AS brand
FROM
  sku_dict_another_one
GROUP BY
  vendor
ORDER BY
  brand DESC,
  vendor ASC
LIMIT
  10; */

SELECT
  vendor,
  COUNT(sku_type) AS sku
FROM
  sku_dict_another_one
GROUP BY
  vendor
ORDER BY
  sku DESC,
  vendor ASC
LIMIT
  10;