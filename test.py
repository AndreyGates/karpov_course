from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

import os
os.environ['JAVA_HOME'] = 'C:\Program Files\Java\jdk-21'

# Create SparkSession
spark = SparkSession.builder.getOrCreate()

# Create DataFrame
data = [("Alice", "2021-01-15"), ("Bob", "2021-02-20"), ("Charlie", "2021-03-10")]
df = spark.createDataFrame(data, ["Name", "Date"])

df.printSchema()
df.show(10)

# Find the most recent date from the "Date" column
#recent_date = df.select(to_date(col("Date")).alias("Date")).selectExpr("max(Date)").first()[0]

#print("Most recent date:", recent_date)