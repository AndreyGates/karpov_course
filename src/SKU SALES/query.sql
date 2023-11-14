select cal_date as days, sum(cnt)::numeric as sku
from transactions_another_one
group by days;