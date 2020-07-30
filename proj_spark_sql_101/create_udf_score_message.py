from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from afinn import Afinn

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("GeoData") \
        .getOrCreate()

    # Define and register UDF
    model = Afinn()

    def score_message_py(msg):
        global model
        return model.score(msg)

    score_message= udf(score_message_py, FloatType()) #FloatType return type# for DataFrame transformation API
    spark.udf.register("score_message", score_message_py, StringType()) # for SQL


    # Get average user sentimental scores
    df = spark.read.csv("hdfs://devenv/user/spark/spark_sql_101/messages/data",
                               header=True,
                               inferSchema=True)
 
    # (i) Using DF transformations
    scored = df.select(df["user"], score_message(df["message"]).alias("score"))
    result = scored.groupBy("user").agg(avg("score"))
    result.show(10000)

    # (ii) Using SQL
    df.createOrReplaceTempView("messages")
    result = spark.sql("select user, avg(score_message(message)) from messages group by user")
    result.show(10000)