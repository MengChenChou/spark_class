from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import VectorAssembler,VectorIndexer,OneHotEncoder,StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .getOrCreate()

    # Prepare data
    final_data = spark.read.csv("hdfs://devenv/user/spark/spark_mllib/datasets/movielens/ratings.csv",
                                inferSchema=True,
                                header=True)


    # Split data into train and test sets
    train_data, test_data = final_data.randomSplit([0.8,0.2])
    
    # Model training
    als = ALS(maxIter=5,userCol="userId",itemCol="movieId",ratingCol="rating")
    model = als.fit(train_data)
    
    # Transform the test data using the model to get predictions
    predicted_test_data = model.transform(test_data)

    # Specify the number of movies you would like to recommand for each user
    user_movies = model.recommendForAllUsers(3)
    user_movies.show()

    # The users who are most likely to like a particular movie
    movie_uers = model.recommendForAllItems(3)
    movie_uers.show()