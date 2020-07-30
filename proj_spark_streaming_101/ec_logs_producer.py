from kafka import KafkaProducer
from random import randint
import time
from datetime import datetime

if __name__ == "__main__":

    rate = 10
    products = 100
    referers = 10
    pages = 200
    visitors = 200
    topic = "logs_stream"
    host_port = "devenv:9092"

    # Create Kafka producer
    producer = KafkaProducer(bootstrap_servers='devenv:9092')

    # Initialization
    ts = time.time()
    actions = ["page_view", "add_to_cart", "sale"]

    # Send data
    while True:
        print("Sending data...")
        for i in range(rate):

            time_field = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            ts = ts + randint(1, 6000)
            referer_field = "referer-" + str(randint(1, referers))
            action_field = actions[randint(0, 2)]
            visitor_field = "visitor-" + str(randint(1, visitors))
            page_field = "page-" + str(randint(1, pages))
            product_field = "product-" + str(randint(1, products))

            message = "{},{},{},{},{},{}".format(time_field, referer_field, action_field, visitor_field, page_field, product_field)
            producer.send("logs_stream", value=bytes(message, "utf8"))

        time.sleep(1)
