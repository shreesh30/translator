import json
import logging
import time

import pika

from src.utils.heartbeat_thread import HeartbeatThread
from src.utils.utils import Utils


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
        self.heartbeat_thread = None

    def connect(self):
        """
        Connects to RabbitMQ and declares the queue.
        """
        while True:
            try:
                credentials = pika.PlainCredentials(Utils.KEY_USER, Utils.KEY_PASSWORD)
                params = pika.ConnectionParameters(host=self.host, credentials=credentials, heartbeat=60, blocked_connection_timeout=1800)
                self.connection = pika.BlockingConnection(params)
                self.channel = self.connection.channel()
                self.channel.queue_declare(queue=self.queue, durable=self.durable)
                logging.info(f"[Producer] Connected to RabbitMQ at {self.host}, queue={self.queue}")

                # Start heartbeat thread
                if self.heartbeat_thread is None:
                    self.heartbeat_thread = HeartbeatThread(self.connection)
                    self.heartbeat_thread.start()

                return
            except Exception as e:
                logging.error(f"[Producer] Connection failed: {e}")
                time.sleep(5)

    def publish(self, message, persistent=True):
        if not self.channel or self.channel.is_closed:
            logging.info("Reconnecting producer...")
            self.connect()

        if isinstance(message, dict):
            message = json.dumps(message)

        properties = pika.BasicProperties(
            delivery_mode=2 if persistent else 1
        )

        self.channel.basic_publish(
            exchange='',
            routing_key=self.queue,
            body=message,
            properties=properties
        )
        logging.info(f"[Producer] Message published to {self.queue}")

    def get_queue_info(self, queue):
        if not self.channel:
            raise RuntimeError("RabbitMQ connection not established. Call connect() first.")

        queue_info= self.channel.queue_declare(queue=queue, passive=True)
        return queue_info

    def close(self):
        """
        Closes the connection to RabbitMQ.
        """
        if self.heartbeat_thread:
            self.heartbeat_thread.stop()
        if self.connection and self.connection.is_open:
            self.connection.close()
            logging.info("[Producer] Connection closed")