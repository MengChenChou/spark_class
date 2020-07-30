from pyspark.sql import SparkSession

if __name__ == "__main__":
    # Create SparkSession
    spark = SparkSession \
        .builder \
        .getOrCreate()

    # Create a DataFrame from a CSV file without header
    stocks_df = spark.read.csv("hdfs://devenv/user/spark/spark_sql_101/data/stocks.txt")

    #  Print the schema in a tree format
    stocks_df.printSchema()

    #  Displays the content of the DataFrame to stdout
    stocks_df.show()

    # Create a DataFrame from a CSV file with header
    stocks_df2 = spark.read.csv("hdfs://devenv/user/spark/spark_sql_101/data/stocks_header.txt", header=True)

    #  Print the schema in a tree format
    stocks_df2.printSchema()

    #  Displays the content of the DataFrame to stdout
    stocks_df2.show()

