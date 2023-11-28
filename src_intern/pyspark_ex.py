# Setting ENV var for Java
import os
os.environ['JAVA_HOME'] = 'C:\Program Files\Java\jdk-21'

# Import the PySpark libraries
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, SQLContext

# Create a SparkSession
spark = SparkSession.builder.appName("MyApp").getOrCreate()

# Create a SQLContext
sqlContext = SQLContext(spark)

# Load a CSV file
df = sqlContext.read.csv("test.csv", header=True)

# Count the number of rows in the DataFrame
rowCount = df.count()

# Print the row count
print("Number of rows:", rowCount)

# Print the schema of the DataFrame
df.printSchema()

# Show the first 10 rows of the DataFrame
df.show(10)
