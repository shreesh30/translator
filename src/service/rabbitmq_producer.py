import json
import logging

import pika


class RabbitMQProducer:
    def __init__(self, queue,host='localhost', durable=True):
        """
        Initialize the producer with connection details.
        :param host: RabbitMQ hostname or IP.
        :param queue: Queue name to publish messages to.
        :param durable: If True, queue survives broker restarts.
        """
        self.host = host
        self.queue = queue
        self.durable = durable
        self.connection = None
        self.channel = None

    def connect(self):
        """
        Connects to RabbitMQ and declares the queue.
        """
        try:
            params = pika.ConnectionParameters(host=self.host)
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue, durable=self.durable)
            logging.info(f"[Producer] Connected to RabbitMQ at {self.host}, queue={self.queue}")
        except Exception as e:
            logging.error(f"[Producer] Connection failed: {e}")
            raise

    def publish(self, message, persistent=True):
        """
        Publishes a message to the queue.
        :param message: Can be a dict or string.
        :param persistent: If True, message survives broker restarts.
        """
        if not self.channel:
            raise RuntimeError("RabbitMQ connection not established. Call connect() first.")

        if isinstance(message, dict):
            message = json.dumps(message)

        properties = pika.BasicProperties(
            delivery_mode=2 if persistent else 1  # 2 = persistent, 1 = transient
        )

        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue,
            body=message,
            properties=properties
        )
        logging.info(f"[Producer] Message published to {self.queue}")

    def close(self):
        """
        Closes the connection to RabbitMQ.
        """
        if self.connection:
            self.connection.close()
            logging.info("[Producer] Connection closed")