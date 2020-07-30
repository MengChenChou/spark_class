from pyspark.sql import SparkSession

if __name__ == "__main__":
    # Create SparkSession
    spark = SparkSession \
        .builder \
        .getOrCreate()

    # Create a DataFrame from a JSON file
    stocks_df = spark.read.json("hdfs://devenv/user/spark/spark_sql_101/data/stocks.json", schema="adj_close float, close float, date date, high float, low float, open float, symbol string, volume long")

    #  Print the schema in a tree format
    stocks_df.printSchema()

    #  Displays the content of the DataFrame to stdout
    stocks_df.show()

    # Register the DataFrame as a SQL temporary view
    stocks_df.createOrReplaceTempView("stocks")

    # use SQL to query teh DataFrame with spark.sql() methods
    result_df = spark.sql("SELECT symbol, AVG(open) as avg_open FROM stocks GROUP BY symbol")

    result_df.show()

    # write to hdfs in parquet file formats
    result_df.write.parquet("hdfs://devenv/user/spark/spark_sql_101/data/stocks_in_parquet")

