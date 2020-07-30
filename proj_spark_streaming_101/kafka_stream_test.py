from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils

if __name__ == "__main__":

    sc = SparkContext()
    ssc = StreamingContext(sc, 1)

    raw_stream = KafkaUtils.createStream(ssc, "devenv:2181", "consumer-group", {"test_stream": 3})
    raw_stream.pprint()

    # Start it
    ssc.start()
    ssc.awaitTermination()
