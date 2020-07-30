from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import re

def logPattern():
    threeDigits = "\\d{1,3}"
    ip = "({}\\.{}\\.{}\\.{})?".format(threeDigits, threeDigits, threeDigits, threeDigits)
    client = "(\\S+)"
    uid = "(\\S+)"
    dateTime = "(\\[.+?\\])"
    request = "\"(.*?)\""
    statusCode = "(\\d{3})"
    bytes = "(\\S+)"
    referer = "\"(.*?)\""
    agent = "\"(.*?)\""
    regex = "{} {} {} {} {} {} {} {} {}".format(ip, client, uid, dateTime, request, statusCode, bytes, referer, agent)
    return re.compile(regex)


if __name__ == "__main__":
    sc = SparkContext()
    ssc = StreamingContext(sc, 1)
    # checkpoint dir is needed when reduceByKeyAndWindow is used
    ssc.checkpoint("hdfs://devenv/user/spark/spark_streaming_101/checkpoint")

    pattern = logPattern()

    lines = ssc.socketTextStream("devenv", 9999)

    def extract_request(line):
        match = pattern.match(line)
        if match:
            return match.group(5)
        else:
            return "InvalidLogFound"

    requests = lines.map(extract_request)

    def extract_url(request):
        arr = request.split(" ")
        if len(arr) == 3:
            return arr[1]
        else:
            return "InvalidLogFound"

    urls = requests.map(extract_url)
    # use inverse operation with reduceByKeyAndWindow
    url_visits = urls.map(lambda x: (x, 1)).reduceByKeyAndWindow(lambda x, y: x + y, lambda x, y: x - y, 600, 10)
    sorted = url_visits.transform(lambda rdd: rdd.sortBy(lambda x: x[1], False))

    sorted.pprint(20)

    ssc.start()
    ssc.awaitTermination()