from pyspark import SparkConf, SparkContext

def is_good(record):
    try:
        temp = int(record.split(",")[10])
    except ValueError:
        return False
    return True


if __name__ == "__main__":
    sc = SparkContext()

    records = sc.textFile("hdfs://devenv/user/spark/spark101/avg_temperature/data")

    good_records = records.filter(is_good)

    day_temp = good_records.map(lambda x: (x.split(",")[1], int(x.split(",")[10])))

    result = day_temp.combineByKey(lambda v: (v, 1), lambda acc, v: (acc[0] + v, acc[1] + 1),
                                   lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])) \
        .map(lambda x: (x[0], x[1][0] / x[1][1]))

    for line in result.collect():
        print(line)
