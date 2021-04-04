from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
import pickle
import cdsw
import os
import time

spark = SparkSession.builder \
      .appName("Predictive Maintenance") \
      .getOrCreate()

# read 21 colunms large file from HDFS
schemaData = StructType([StructField("0", DoubleType(), True),
                         StructField("1", DoubleType(), True),
                         StructField("2", DoubleType(), True),
                         StructField("3", DoubleType(), True),
                         StructField("4", DoubleType(), True),
                         StructField("5", DoubleType(), True),
                         StructField("6", DoubleType(), True),
                         StructField("7", DoubleType(), True),
                         StructField("8", DoubleType(), True),
                         StructField("9", DoubleType(), True),
                         StructField("10", DoubleType(), True),
                         StructField("11", DoubleType(), True),
                         StructField("12", IntegerType(), True)])

iot_data = spark.read.schema(schemaData).csv('/user/'
                                             + os.environ['HADOOP_USER_NAME']
                                             + '/historical_iot.txt')

# Inspect the Data
iot_data.show()
iot_data.printSchema()

#Put Data to Hive Table
spark.sql("show databases").show()
spark.sql("show tables in default").show()

# Create the Hive table
# This is here to create the table in Hive used be the other parts of the project, if it
# does not already exist.

if ('historical_iot' not in list(spark.sql("show tables in default").toPandas()['tableName'])):
  print("creating the historical_iot table")
  iot_data\
    .write.format("parquet")\
    .mode("overwrite")\
    .saveAsTable(
      'default.historical_iot'
  )

# Show the data in the hive table
spark.sql("select * from default.historical_iot").show()

# To get more detailed information about the hive table you can run this:
spark.sql("describe formatted default.historical_iot").toPandas()