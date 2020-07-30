from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler,VectorIndexer,OneHotEncoder,StringIndexer
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .getOrCreate()

    # Prepare data
    raw = spark.read.csv("hdfs://devenv/user/spark/spark_mllib1/titanic",
                          inferSchema=True,
                          header=True)

    # Preprocessing and feature engineering
    data = raw.select("Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked") \
              .dropna()

    data = StringIndexer(inputCol="Sex",outputCol="SexIndex").fit(data).transform(data)

    data = OneHotEncoder(inputCol="SexIndex",outputCol="SexVec").transform(data)

    data = StringIndexer(inputCol="Embarked",outputCol="EmbarkIndex").fit(data).transform(data)

    data = OneHotEncoder(inputCol="EmbarkIndex",outputCol="EmbarkVec").transform(data)

    final_data = VectorAssembler(inputCols=["Pclass","SexVec","Age","SibSp","Parch","Fare","EmbarkVec"],
                                 outputCol="features").transform(data)

    # Split data into train and test sets
    train_data, test_data = final_data.randomSplit([0.7,0.3])
    
    # Model training
    classifier = LogisticRegression(featuresCol="features",labelCol="Survived")
    model = classifier.fit(train_data)
    
    # Transform the test data using the model to get predictions
    predicted_test_data = model.transform(test_data)

    # Evaluate the model performance
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol='Survived',
                                                     predictionCol='prediction', 
                                                     metricName='f1')
    print("F1 score: {}", evaluator_f1.evaluate(predicted_test_data))

    evaluator_accuracy = MulticlassClassificationEvaluator(labelCol='Survived',
                                                           predictionCol='prediction', 
                                                           metricName='accuracy')
    print("Accuracy: {}", evaluator_accuracy.evaluate(predicted_test_data)) 


    # Predict some new records
    # In real case, use VectorAssembler to transform df for features column
    data_to_predict = final_data.select("features").limit(10) 
    model.transform(data_to_predict).show()

    # Save the model
    model.save("/home/spark/Desktop/logistic_regression_model_titanic")

    # Read the saved model
    model_reloaded = LogisticRegressionModel.load("/home/spark/Desktop/logistic_regression_model_titanic") 

    # Predict some new records
    # In real case, use VectorAssembler to transform df for features column
    data_to_predict = final_data.select("features").limit(10) 
    model_reloaded.transform(data_to_predict).show()