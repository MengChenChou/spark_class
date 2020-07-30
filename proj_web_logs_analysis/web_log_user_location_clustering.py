from pyspark.sql import SparkSession
from pyspark.sql.functions import count, avg, col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .getOrCreate()

    # Prepare data
    logs = spark.read.csv("hdfs://devenv/user/spark/web_logs_analysis/data/", header=True, inferSchema=True)

    # Precessing and feature engineering
    all_users = logs.select("device_id") \
        .distinct() \
        .rdd.map(lambda row: row[0]).collect()

    
    num_users_to_predict = 5 # only for demo purpose for not running all users

    for user in all_users:
        user_locations = logs.select("device_id", "lat", "lon").filter("device_id = '{}'".format(user))

        user_data = VectorAssembler(inputCols=["lat", "lon"],
                                    outputCol="features").transform(user_locations)
        # Model training
        kmeans = KMeans(k=5)
        model = kmeans.fit(user_data)

        # Transform the test data using the model to get predictions
        clustered_data = model.transform(user_data)
       
        # Prediction and model status
        clustered_data_sorted = clustered_data.orderBy("prediction")
        
        print("### User: {}".format(user))

        clustered_data_sorted.show(20)

        # Cluster centers and count
        clustered_data.groupBy("prediction") \
                      .agg(avg("lat"), avg("lon"), count("*").alias("location_count")) \
                      .orderBy("prediction", col("location_count").desc()) \
                      .show()

        centers = model.clusterCenters()
        print("Cluster Centers: ")
        for center in centers:
            print(center)

        num_users_to_predict -= 1
        if num_users_to_predict == 0:
            break
