from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from afinn import Afinn

if __name__ == "__main__":

    sc = SparkContext()
    ssc = StreamingContext(sc, 10)

    # A sentiment analysis model
    model = Afinn()
    # Messages to print
    print_message_number = 1000000

    raw_stream = KafkaUtils.createStream(ssc, "devenv:2181", "consumer-group", {"tweets_stream": 1}) \
                           .window(300, 20)

    raw_stream.mapValues(lambda message: (model.score(message), 1)) \
              .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
              .mapValues(lambda x: x[0] / x[1]) \
              .transform(lambda rdd: rdd.sortBy(lambda x: x[1], ascending=False)) \
              .pprint(print_message_number)

    # Start it
    ssc.start()
    ssc.awaitTermination()