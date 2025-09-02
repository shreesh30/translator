import json
import logging
import time

import pika
from pika.exceptions import AMQPConnectionError

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
        while True:
            try:
                credentials = pika.PlainCredentials(Utils.KEY_USER, Utils.KEY_PASSWORD)
                params = pika.ConnectionParameters(host=self.host, credentials=credentials, heartbeat=120,blocked_connection_timeout=300)
                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=self.queue, durable=self.durable)
                self.channel.basic_qos(prefetch_count=self.prefetch_count)
                logging.info(f"[Consumer] Connected to RabbitMQ at {self.host}")
                return
            except Exception as e:
                logging.error(f"[Consumer] Connection failed: {e}. Retrying in 5s...")
                time.sleep(5)

    def consume(self, callback, auto_ack=False):
        """
        Start consuming messages.
        :param callback: Function to process each message.
                         Signature: callback(ch, method, properties, body)
        :param auto_ack: If True, messages are auto-acknowledged.
        """
        while True:
            try:
                if not self.connection or self.connection.is_closed:
                    self.connect()

                self.channel.basic_consume(
                    queue=self.queue,
                    on_message_callback=callback,
                    auto_ack=auto_ack
                )
                logging.info(f"[Consumer] Waiting for messages on {self.queue}...")

                self.channel.start_consuming()

            except AMQPConnectionError as e:
                logging.error(f"[Consumer] Lost connection: {e}, reconnecting...")
                time.sleep(5)
            except Exception as e:
                logging.error(f"[Consumer] Unexpected error: {e}, reconnecting...")
                time.sleep(5)

    def close(self):
        """
        Closes the connection to RabbitMQ.
        """
        if self.connection:
            self.connection.close()
            logging.info("[Consumer] Connection closed")