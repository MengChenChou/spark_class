from pyspark import SparkContext
import re


def log_pattern():
    three_digits = "\\d{1,3}"
    ip = "({}\\.{}\\.{}\\.{})?".format(three_digits, three_digits, three_digits, three_digits)
    client = "(\\S+)"
    uid = "(\\S+)"
    date_time = "(\\[.+?\\])"
    request = "\"(.*?)\""
    status_code = "(\\d{3})"
    bytes_part = "(\\S+)"
    referer = "\"(.*?)\""
    agent = "\"(.*?)\""
    regex = "{} {} {} {} {} {} {} {} {}".format(ip, client, uid, date_time, request, status_code, bytes_part, referer, agent)
    return re.compile(regex)


def extract_ip(line):
    match = pattern.match(line)
    if match:
        return match.group(1)
    else:
        return "InvalidLogFound"


if __name__ == "__main__":
    sc = SparkContext()

    pattern = log_pattern()

    lines = sc.textFile("hdfs://devenv/user/spark/spark101/access_log_analysis/data")

    ips = lines.map(extract_ip)

    ip_visits = ips.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)

    ip_visits_sorted = ip_visits.sortBy(lambda x: x[1], False)

    result = ip_visits_sorted.collect()

    for ip, count in result:
        print("{}: {}".format(ip, count))