from kafka import KafkaProducer
import time
import csv

if __name__ == "__main__":
    # Configurationss
    rate = 10
    topic = "tweets_stream"
    bootstrap_servers = "devenv:9092,devenv:9093"

    # Create Kafka producer
    producer = KafkaProducer(bootstrap_servers=bootstrap_servers)

    # open CSV file containing users' messages
    with open("/home/spark/Desktop/spark_streaming_101/simple_sentiment_analysis_demo/data/messages.csv") as csvfile:
        # initialize counters
        batch_message_sent_count = 0
        error_count = 0

        # open CVS file
        rows = csv.reader(csvfile)

        # Send messages
        for row in rows:
            try:
                user = row[4]
                message = row[5]
                producer.send(topic, key=bytes(user, "utf8"), value=bytes(message, "utf8"))
            except:
                error_count += 1

            batch_message_sent_count += 1

            if rate == batch_message_sent_count:
                batch_message_sent_count = 0
                time.sleep(1)
