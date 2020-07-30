from pyspark import SparkContext
import re
from geoip2.database import Reader


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


def extract_location(line):
    global reader

    if reader is None:
        reader = Reader("/home/spark/spark-2.4.5-bin-hadoop2.7/maxmind/GeoLite2-City.mmdb")

    match = pattern.match(line)

    if match:
        ip = match.group(1)
        response = reader.city(ip)
        country = response.country.name
        city = response.city.name

        if city is None:
            return country
        else:
            return "{},{}".format(country, city)

    else:
        return "InvalidLogFound"


if __name__ == "__main__":
    sc = SparkContext()

    pattern = log_pattern()

    reader = None

    lines = sc.textFile("hdfs://devenv/user/spark/spark101/access_log_analysis/data")

    ips = lines.map(extract_location)

    location_visits = ips.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)

    location_visits_sorted = location_visits.sortBy(lambda x: x[1], False)

    result = location_visits_sorted.collect()

    for ip, count in result:
        print("{}: {}".format(ip, count))
