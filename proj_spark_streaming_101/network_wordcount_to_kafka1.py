from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from kafka import KafkaProducer
import re

def output_rdd(rdd):
    # Transfer the rdd's data back to the Driver
    rdd_data = rdd.collect()

    # Create producer
    producer = KafkaProducer(bootstrap_servers=broker_list)
    # Get each (word,count) pair and send it to the topic by iterating the partition (an Iterable object)
    for word, count in rdd_data:
        message = "{},{}".format(word, str(count))
        producer.send(topic, value=bytes(message, "utf8"))

    producer.close()


if __name__ == "__main__":

    topic = "wordcount_result"
    broker_list = 'devenv:9092,devenv:9093'

    # Create a local StreamingContext with two working thread and batch interval of 1 second
    sc = SparkContext()
    ssc = StreamingContext(sc, 5)

    # Create a DStream that will connect to hostname:port, like localhost:9999
    lines = ssc.socketTextStream("devenv", 9999)

    # Split each line into words
    words = lines.flatMap(lambda line: re.compile(r"\W").split(line.lower()))

    # Count each word in each batch
    pairs = words.map(lambda word: (word, 1))
    word_counts = pairs.reduceByKey(lambda x, y: x + y)

    word_counts.foreachRDD(output_rdd)

    ssc.start()             # Start the computation
    ssc.awaitTermination()  # Wait for the computation to terminate
