import json
import logging
import pika

from src.utils.utils import Utils


class RabbitMQConsumer:
    def __init__(self, queue, host='localhost', durable=True, prefetch_count=1):
        """
        Initialize the consumer with connection details.
        :param host: RabbitMQ hostname or IP.
        :param queue: Queue name to consume messages from.
        :param durable: If True, queue survives broker restarts.
        :param prefetch_count: How many messages to prefetch per worker.
        """
        self.host = host
        self.queue = queue
        self.durable = durable
        self.prefetch_count = prefetch_count
        self.connection = None
        self.channel = None

    def connect(self):
        """
        Connects to RabbitMQ and declares the queue.
        """
        try:
            credentials = pika.PlainCredentials(Utils.KEY_USER, Utils.KEY_PASSWORD)
            params = pika.ConnectionParameters(host=self.host, credentials=credentials, heartbeat=120,blocked_connection_timeout=300)
            self.connection = pika.BlockingConnection(params)
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue, durable=self.durable)
            self.channel.basic_qos(prefetch_count=self.prefetch_count)
            logging.info(f"[Consumer] Connected to RabbitMQ at {self.host}, queue={self.queue}")
        except Exception as e:
            logging.error(f"[Consumer] Connection failed: {e}")
            raise

    def consume(self, callback, auto_ack=False):
        """
        Start consuming messages.
        :param callback: Function to process each message.
                         Signature: callback(ch, method, properties, body)
        :param auto_ack: If True, messages are auto-acknowledged.
        """
        if not self.channel:
            raise RuntimeError("RabbitMQ connection not established. Call connect() first.")

        self.channel.basic_consume(
            queue=self.queue,
            on_message_callback=callback,
            auto_ack=auto_ack
        )
        logging.info(f"[Consumer] Waiting for messages on {self.queue}...")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logging.info("[Consumer] Stopped manually")
            self.close()

    def close(self):
        """
        Closes the connection to RabbitMQ.
        """
        if self.connection:
            self.connection.close()
            logging.info("[Consumer] Connection closed")