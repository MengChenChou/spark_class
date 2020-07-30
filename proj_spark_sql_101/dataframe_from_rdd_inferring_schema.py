from pyspark.sql import SparkSession, Row

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .getOrCreate()

    sc = spark.sparkContext

    # Load a text file and convert each line to a Row.
    lines = sc.textFile("hdfs://devenv/user/spark/spark_sql_101/data/people.txt")
    parts = lines.map(lambda l: l.split(","))
    people = parts.map(lambda p: Row(name=p[0], age=int(p[1])))

    # Infer the schema, and register the DataFrame as a table.

    # (i) use spark.createDataFrame()
    schemaPeople = spark.createDataFrame(people)

    schemaPeople.printSchema()
    schemaPeople.show()

    # (ii) use people.toDF()
    schemaPeople = people.toDF()

    schemaPeople.printSchema()
    schemaPeople.show()

    # SQL can be run over DataFrames that have been registered as a table.
    schemaPeople.createOrReplaceTempView("people")
    teenagers = spark.sql("SELECT name, age FROM people WHERE age >= 13 AND age <= 19")

    teenagers.show()

    # collect() is a DataFrame action that return a list of Rows
    for teenName in teenagers.collect():
        print(teenName)

