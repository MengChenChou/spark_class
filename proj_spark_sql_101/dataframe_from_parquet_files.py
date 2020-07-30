from pyspark.sql import SparkSession

if __name__ == "__main__":
    # Create SparkSession
    spark = SparkSession \
        .builder \
        .getOrCreate()

    # Create a DataFrame from a folder containing parquet files
    stocks_df = spark.read.parquet("hdfs://devenv/user/spark/spark_sql_101/data/stocks_in_parquet")

    #  Print the schema in a tree format
    stocks_df.printSchema()

    #  Displays the content of the DataFrame to stdout
    stocks_df.show()

