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


    spark.udf.register("get_geo_locations", get_locations_py, ArrayType(StringType()))

    report_input = spark.read.csv("hdfs://devenv/user/spark/audi_case_study/location_info_added",
                                  schema="""id INT, create_time TIMESTAMP, action_time TIMESTAMP, log_type INT, 
                                  ad_id STRING, position_method INT, location_accuracy FLOAT, lat DOUBLE, lon DOUBLE, 
                                  cell_id STRING, lac STRING, mcc STRING, mnc STRING, ip STRING, connection_type INT, 
                                  imei STRING, android_id STRING, udid STRING, open_udid STRING, idfa STRING, 
                                  mac_address STRING, uid STRING, density FLOAT, screen_height INT, screen_width INT, 
                                  ua STRING, app_id STRING, model_id STRING, carrier_id STRING, os_id INT, os_ver STRING, 
                                  ip_country STRING, ip_city STRING, ip_lat DOUBLE, ip_lon DOUBLE, ip_quadkey STRING""")
   
    report_input.createOrReplaceTempView("report_input")

    result = spark.sql("""
                        SELECT ad_id, region,SUM(imp),COUNT(DISTINCT imp_imei),SUM(click),COUNT(DISTINCT click_imei)
                        FROM 
                        (
                            SELECT
                            ad_id,
                            region,
                            CASE WHEN log_type=1 THEN 1 ELSE 0 END AS imp,
                            CASE WHEN log_type=1 THEN imei ELSE null END AS imp_imei,
                            CASE WHEN log_type=2 THEN 1 ELSE 0 END AS click,
                            CASE WHEN log_type=2 THEN imei ELSE null END AS click_imei,
                            explode(get_geo_locations(ip_quadkey)) AS region
                            FROM report_input
                        ) src
                        GROUP BY ad_id, region
                        ORDER BY ad_id, region""")

    result.show(1000000, truncate=False)
