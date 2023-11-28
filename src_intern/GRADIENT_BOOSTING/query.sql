SELECT age, income, dependents,
       has_property, has_car, credit_score,
       job_tenure, has_education, loan_amount,
       DATEDIFF(day, loan_start, loan_deadline) AS loan_period, 
       greatest(DATEDIFF(day, loan_deadline, loan_payed), 0) AS delay_days
FROM default.loan_delay_days
ORDER BY id;
