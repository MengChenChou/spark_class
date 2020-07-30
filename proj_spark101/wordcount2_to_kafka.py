import re
from pyspark import SparkContext
from kafka import KafkaProducer

def output_partition_to_kafka(partition):
    # Create producer
    producer = KafkaProducer(bootstrap_servers="localhost:9092)
    # Get each (word,count) pair and send it to the topic by iterating the partition (an Iterable object)
    for word, count in partition:
        message = "{},{}".format(word, str(count))
        producer.send("test_stream", value=bytes(message, "utf8"))

    producer.close()


if __name__ == "__main__":
    sc = SparkContext()

    lines = sc.textFile("hdfs://devenv/user/spark/spark101/wordcount/data")

    words = lines.flatMap(lambda x: re.compile(r'\W+', re.UNICODE).split(x.lower()))

    word_counts = words.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
  
    word_counts.foreachPartition(output_partition_to_kafka)
