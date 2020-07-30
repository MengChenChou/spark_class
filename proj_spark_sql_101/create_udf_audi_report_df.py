from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from quadkey_template_db import QuadkeyTemplateDB


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("GeoData") \
        .getOrCreate()

    qkDB = None

    def get_locations_py(quadkey):
        global qkDB
        if qkDB is None:
            qkDB = QuadkeyTemplateDB("./qk_cn.csv")
        found = qkDB.lookup_regions(quadkey)
        if len(found) != 0 or found is None:
            return found
        else:
            return ["NotFound"]


    get_locations = udf(get_locations_py, ArrayType(StringType()))

    report_input = spark.read.csv("hdfs://devenv/user/spark/audi_case_study/location_info_added",
                                  schema="""id INT, create_time TIMESTAMP, action_time TIMESTAMP, log_type INT, 
                                  ad_id STRING, position_method INT, location_accuracy FLOAT, lat DOUBLE, lon DOUBLE, 
                                  cell_id STRING, lac STRING, mcc STRING, mnc STRING, ip STRING, connection_type INT, 
                                  imei STRING, android_id STRING, udid STRING, open_udid STRING, idfa STRING, 
                                  mac_address STRING, uid STRING, density FLOAT, screen_height INT, screen_width INT, 
                                  ua STRING, app_id STRING, model_id STRING, carrier_id STRING, os_id INT, os_ver STRING, 
                                  ip_country STRING, ip_city STRING, ip_lat DOUBLE, ip_lon DOUBLE, ip_quadkey STRING""")

    result = report_input.select("ad_id", explode(get_locations("ip_quadkey")).alias("region"), "log_type", "imei") \
        .withColumn("imp", when(report_input["log_type"] == 1, 1).otherwise(0)) \
        .withColumn("imp_imei", when(report_input["log_type"] == 1, report_input["imei"]).otherwise(None)) \
        .withColumn("click", when(report_input["log_type"] == 2, 1).otherwise(0)) \
        .withColumn("click_imei", when(report_input["log_type"] == 2, report_input["imei"]).otherwise(None)) \
        .groupBy("ad_id", "region") \
        .agg(sum("imp").alias("imp_count"), countDistinct("imp_imei").alias("imp_uu_count"),
             sum("click").alias("click_count"),countDistinct("click_imei").alias("click_uu_count")) \
        .orderBy("ad_id","region")

    result.show(1000000, truncate=False)
