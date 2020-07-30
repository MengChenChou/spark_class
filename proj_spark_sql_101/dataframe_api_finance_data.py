from pyspark.sql import SparkSession
from pyspark.sql.functions import *

if __name__ == "__main__":
    spark = SparkSession \
            .builder \
            .appName("Finance Data") \
            .getOrCreate()

    finances_df = spark.read.json("hdfs://devenv/user/spark/spark_sql_101/data/finances.json")


    etl_df = finances_df.na.drop("all", subset=["ID", "Account", "Amount", "Description", "Date"]) \
                        .na.fill("Unknown", "Description") \
                        .where((finances_df["Amount"] != 0) | (finances_df["Description"] == "Unknown")) \
                        .select("Account.Number", "Amount", "Date", "Description")
    etl_df.show()


    corrupt_df = finances_df.where(finances_df["_corrupt_record"].isNotNull()).select("_corrupt_record", "ID")
    corrupt_df.show()

    account_df = finances_df.select(concat("Account.FirstName",lit(" "),"Account.LastName").alias("FullName"),
                                           finances_df["Account.Number"].alias("AccountNumber")) \
                            .na.drop("all") \
                            .distinct()
    account_df.show()
    
    account_details_df = finances_df.select(finances_df["Account.Number"].alias("AccountNumber"),
                                            finances_df["Amount"],
                                            finances_df["Description"],
                                            to_date(unix_timestamp(finances_df["Date"], "MM/dd/yyyy").cast("timestamp")).alias("Date")) \
                                     .na.drop("all", subset=["AccountNumber", "Amount", "Description", "Date"]) \
                                     .groupBy("AccountNumber") \
                                     .agg(avg("Amount").alias("AverageTransaction"), sum("Amount").alias("TotalTransactions"),
                                          count("Amount").alias("NumberOfTransactions"), max("Amount").alias("MaxTransaction"),
                                          min("Amount").alias("MinTransaction"), stddev("Amount").alias("StandardDeviationAmount"),
                                          collect_set("Description").alias("UniqueTransactionDescriptions"))

    account_details_df.show(truncate=False)
