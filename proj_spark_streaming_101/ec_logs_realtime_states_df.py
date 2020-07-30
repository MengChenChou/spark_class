from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import *
from kafka import KafkaProducer


def rdd_stats(rdd):
    if(rdd.count() == 0):
        return rdd

    input_df = rdd.map(lambda line: line.split(",")) \
                  .map(lambda arr: Row(timestamp=arr[0],
                                       referrer=arr[1],
                                       action=arr[2],
                                       visitor=arr[3],
                                       page=arr[4],
                                       product=arr[5])) \
                  .toDF()

    visitor_activity = input_df.select(input_df["visitor"],
                                       when(input_df["action"] == "sale", 1).otherwise(0).alias("is_sale"),
                                       when(input_df["action"] == "add_to_cart", 1).otherwise(0).alias("is_add_to_cart"),
                                       when(input_df["action"] == "page_view", 1).otherwise(0).alias("is_page_view"))

    user_activity_df = visitor_activity.groupBy("visitor") \
                                       .agg(sum("is_sale").alias("number_of_sales"),
                                            sum("is_add_to_cart").alias("number_of_add_to_carts"),
                                            sum("is_page_view").alias("number_of_page_views"),
                                            count("*").alias("number_of_events"))

    return user_activity_df.rdd.map(lambda row: (row["visitor"], row))


def reduce_function(row1, row2):
    number_of_sales = row1["number_of_sales"] + row2["number_of_sales"]
    number_of_add_to_carts = row1["number_of_add_to_carts"] + row2["number_of_add_to_carts"]
    number_of_page_views = row1["number_of_page_views"] + row2["number_of_page_views"]
    number_of_events = row1["number_of_events"] + row2["number_of_events"]

    return Row(number_of_sales=number_of_sales,
               number_of_add_to_carts=number_of_add_to_carts,
               number_of_page_views=number_of_page_views,
               number_of_events=number_of_events)


def output_partition(partition):
    # Create producer
    producer = KafkaProducer(bootstrap_servers=broker_list)

    for user, row in partition:
        message = "{},{},{},{},{}".format(user,
                                          row["number_of_sales"],
                                          row["number_of_add_to_carts"],
                                          row["number_of_page_views"],
                                          row["number_of_events"])

        producer.send(output_topic, value=bytes(message, "utf8"))

    producer.close()


def output_rdd(rdd):
    rdd.foreachPartition(output_partition)


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .getOrCreate()

    sc = spark.sparkContext
    ssc = StreamingContext(sc, 5)

    broker_list = 'devenv:9092,devenv:9093'
    output_topic = "user_activity_stream"

    raw_stream = KafkaUtils.createStream(ssc, "devenv:2181", "consumer-group", {"logs_stream": 3}) \
                                .map(lambda x: x[1]) \
                                .window(60, 5)

    user_activity_stream = raw_stream.transform(rdd_stats) \
                                     .reduceByKey(reduce_function) \
                                     .transform(lambda rdd: rdd.sortBy(lambda pair: pair[1]["number_of_add_to_carts"],
                                                                       False)) \
                                     .cache()

    # output to console
    user_activity_stream.pprint(20)
    # output to Kafka
    user_activity_stream.foreachRDD(output_rdd)

    # Start it
    ssc.start()
    ssc.awaitTermination()