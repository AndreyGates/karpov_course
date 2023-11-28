select date_trunc('month', date)::date as time,
       mode,
       count(case when status = 'Confirmed' then 1 else null end) * 100.0 / sum(count(*)) over(partition by date_trunc('month', date)::date, mode) as percents
from new_payments
where mode <> 'Не определено'
group by time, mode
order by time, mode;