from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .getOrCreate()

    # Prepare data
    logs = spark.read.csv("hdfs://devenv/user/spark/web_logs_analysis/data/", header=True, inferSchema=True)

    # Preprocessing and feature engineering
    feature_prep = logs.select("product_category_id", "device_type", "connection_type", "gender") \

    final_data = VectorAssembler(inputCols=["product_category_id", "device_type", "connection_type"],
                                 outputCol="features").transform(feature_prep)

    # Split data into train and test sets
    train_data, test_data = final_data.randomSplit([0.7, 0.3])

    # Model training
    classifier = RandomForestClassifier(featuresCol="features", labelCol="gender", numTrees=10, maxDepth=10)
    model = classifier.fit(train_data)

    # Transform the test data using the model to get predictions
    predicted_test_data = model.transform(test_data)

    # Evaluate the model performance
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol='gender',
                                                     predictionCol='prediction',
                                                     metricName='f1')
    print("F1 score: {}", evaluator_f1.evaluate(predicted_test_data))

    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol='gender',
                                                           predictionCol='prediction',
                                                           metricName='accuracy')
    print("Accuracy: {}", evaluator_accuracy.evaluate(predicted_test_data))

    # Predict some new records
    # In real case, use VectorAssembler to transform df for features column
    data_to_predict = final_data.select("features").limit(10)
    model.transform(data_to_predict).show()

    # Save the model
    model.save("hdfs://devenv/user/spark/web_logs_analysis/gender_model/")

    # Read the saved model
    model_reloaded = RandomForestClassificationModel.load("hdfs://devenv/user/spark/web_logs_analysis/gender_model/")

    # Predict some new records
    # In real case, use VectorAssembler to transform df for features column
    data_to_predict = final_data.select("features").limit(10)
    model_reloaded.transform(data_to_predict).show()