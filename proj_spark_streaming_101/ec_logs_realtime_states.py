from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import *
from kafka import KafkaProducer


def output_partition(partition):
    # Create producer
    producer = KafkaProducer(bootstrap_servers=broker_list)

    for e in partition:
        visitor = e[0]
        stats = e[1]

        message = "{},{},{},{},{}".format(visitor,
                                          stats[0],
                                          stats[1],
                                          stats[2],
                                          stats[3])

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


    def extract_numbers_in_pair(line):
        arr = line.split(",")
        visitor = arr[3]
        action = arr[2]

        num_sale = 1 if action == "sale" else 0
        num_add_to_cart = 1 if action == "add_to_cart" else 0
        num_page_view = 1 if action == "page_view" else 0
        num_total = 1

        return visitor, (num_sale, num_add_to_cart, num_page_view, num_total)


    user_activity_stream = raw_stream.map(extract_numbers_in_pair) \
                                     .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2], x[3] + y[3])) \
                                     .transform(lambda rdd: rdd.sortBy(lambda pair: pair[1][1], # order by num_add_to_cart
                                                                       False)) \
                                     .cache()

    # output to console
    user_activity_stream.pprint(20)
    # output to Kafka
    user_activity_stream.foreachRDD(output_rdd)

    # Start it
    ssc.start()
    ssc.awaitTermination()