from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .getOrCreate()
    
    df = spark.read.csv("hdfs://devenv/user/spark/spark_sql_101/data/stocks_header.txt", inferSchema=True,
                        header=True)
    df.printSchema()
    df.show()

    def slen_py(s):
        return len(s)

    spark.udf.register("slen", slen_py, IntegerType())

    df.createOrReplaceTempView("stocks")
    spark.sql("select slen(symbol) as length_of_symbol from stocks").show()


    slen = udf(slen_py, IntegerType())

    df.select(slen("symbol").alias("length_of_symbol")).show()
