from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from geoip2.database import Reader # assume all work nodes have geoip2 installed

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("GeoData") \
        .getOrCreate()

    reader = None

    def ip2city_py(ip):
        global reader
        if reader is None:
            # assume all work nodes have mmdb installed in the following path
            reader = Reader("/home/spark/spark-2.4.5-bin-hadoop2.7/maxmind/GeoLite2-City.mmdb")
        try:
            response = reader.city(ip)
            city = response.city.name
            if city is None:
                return None
            return city

        except:
            return None

    ip2city= udf(ip2city_py, StringType())

    page_view = spark.read.csv("hdfs://devenv/user/spark/spark_sql_101/page_views/data",
                               sep="\t",
                               schema="logtime STRING, userid INT, ip STRING, page STRING, ref STRING, os STRING, os_ver STRING, agent STRING")


    page_view_city = page_view.withColumn("city", ip2city("ip"))
    page_view_city.show()

    stats_by_city_sorted = page_view_city.fillna("unknown", subset=["city"]) \
                                         .groupBy(col("city")) \
                                         .agg(count("*").alias("records"), countDistinct("userid").alias("UU")) \
                                         .orderBy(col("records").desc())
    stats_by_city_sorted.show(10000)
