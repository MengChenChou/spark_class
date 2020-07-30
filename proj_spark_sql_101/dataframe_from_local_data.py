from pyspark.sql import SparkSession, Row

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .getOrCreate()

    people = [(1, "John", 10), (2, "Peter", 15), (3, "Mary", 12)]
    people_df = spark.createDataFrame(people)
    people_df.printSchema()
    people_df.show()

    people_df2 = spark.createDataFrame(people, "id: int, name: string, age: int")
    people_df2.printSchema()
    people_df2.show()

    people2 = [Row(id=1, name="John", age=10),
               Row(id=2, name="Peter", age=15),
               Row(id=3, name="Mary", age=12)]

    people_df3 = spark.createDataFrame(people2)
    people_df3.printSchema()
    people_df3.show()

    people_df4 = spark.createDataFrame(people2, "id: int, name: string, age: int")
    people_df4.printSchema()
    people_df4.show()
