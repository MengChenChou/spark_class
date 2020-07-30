from pyspark.sql import SparkSession
from pyspark.sql.functions import *

if __name__ == "__main__":
    spark = SparkSession \
            .builder \
            .appName("Stocks Data") \
            .getOrCreate()

    df = spark.read.csv("hdfs://devenv/user/spark/spark_sql_101/data/stocks_header.txt",
                        inferSchema=True,
                        header=True)

    df.show()
    df.select("symbol", "high", "low").show()
    df.select(df["symbol"], df["high"], df["low"]).show()
    df.select(df["symbol"].alias("Stock Symbol"), df["high"], df["low"]).show()
    df.select(df["symbol"].alias("Stock Symbol"), format_number(df["high"],1), df["low"]).show()
    df.select("symbol", "high", df["high"] + 10, "low").show()

    df.filter(df["symbol"] == "AAPL").show()
    df.filter((df["close"] <= 200) & (df["open"] > 30)).show()
    df.filter((df["close"] <= 200) & ~(df["open"] > 30)).show()

    df.filter("symbol = 'AAPL'").show()
    df.filter("close <= 200 and open > 30").show()
    df.filter("close <= 200 and not(open > 30)").show()

    df.where("close <= 200 and not(open > 30)").show()

    df.orderBy(df["open"]).show()
    df.orderBy(df["symbol"], df["open"]).show()
    df.orderBy(df["open"].desc()).show()
    df.orderBy("symbol", "open").show()


    stock_names = [("AAPL", "Apple"), ("CSCO","Cisco Systems")]
    stock_names_df = spark.createDataFrame(stock_names, "symbol: string, names: string")

    stock_names_df.join(df, stock_names_df["symbol"] == df["symbol"]).show()
    stock_names_df.join(df, stock_names_df["symbol"] == df["symbol"], "right").show(40)

    df.groupBy(df["symbol"])
    df.groupBy(df["symbol"]).mean().show()
    df.groupBy(df["symbol"]).max().show()

    df.groupBy(df["symbol"]).agg({

        "open": "avg",
        "close": "stddev",
        "high": "max",
        "low" : "min",
        "volumn": "sum"

    }).show()

    df.groupBy(df["symbol"]).agg(avg("open"), stddev("close"), max("high"), min("low"), sum("volumn")).show()

    df.groupBy("symbol").mean().show()

    df.count()
    df.agg({"open": "max", "close": "min"}).show()
    df.agg(countDistinct("symbol")).show()

    df_10 = df.limit(10)
    df_10.show()

    df.select("symbol").distinct().show()

    df.select("*", (df["open"] - df["close"]).alias("new column")).show()


    df.withColumn("new column", df["open"] - df["close"]).show()

    df.withColumnRenamed("symbol", "stock symbol").show()

    df.show()

    df.drop("volumn", "adj_close").show()

    df = spark.read.csv("hdfs://devenv/user/spark/spark_sql_101/data/contains_null.txt", inferSchema=True, header=True)

    df.na.drop().show()
    df.na.drop(how="all").show()
    df.na.drop(how="any").show()
    df.na.drop(how="any", subset=["Name"]).show()
    df.dropna(how="all").show()
    df.dropna(how="any", subset=["Name"]).show()

    df.na.fill("unknown", subset=["Name"]).show()
    df.fillna("unknown", subset=["Name"]).show()

    df = spark.read.csv("hdfs://devenv/user/spark/spark_sql_101/data/stocks_header.txt",
                        inferSchema=True,
                        header=True)

    df.count()
    df.show()

    df.describe().show()
    df.describe("open","close").show()

    df.select(stddev(df["open"]).alias("std_open")).show()
    df.agg(stddev(df["open"]))


    df.select(format_number(stddev(df["open"]),2).alias("std_open")).show()
    df.select(dayofmonth(df["day"])).limit(10).show()
    df.select(dayofyear(df["day"])).limit(10).show()
    df.select(year(df["day"])).limit(10).show()
    df.select(month(df["day"])).limit(10).show()


    df.columns
    df.printSchema()