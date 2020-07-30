from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import re

if __name__ == "__main__":
    # Create a local StreamingContext with two working thread and batch interval of 1 second
    sc = SparkContext()
    ssc = StreamingContext(sc, 1)

    # Create a DStream that will connect to hostname:port, like localhost:9999
    lines = ssc.socketTextStream("devenv", 9999)

    # Split each line into words
    words = lines.flatMap(lambda line: re.compile(r"\W+").split(line.lower()))

    # Count each word in each batch
    pairs = words.map(lambda word: (word, 1))
    word_counts = pairs.reduceByKey(lambda x, y: x + y)

    # Print the first ten elements of each RDD generated in this DStream to the console
    word_counts.pprint(30)

    ssc.start()             # Start the computation
    ssc.awaitTermination()  # Wait for the computation to terminate
