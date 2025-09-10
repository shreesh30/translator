import logging
import time

import pika
from pika.exceptions import AMQPConnectionError, StreamLostError

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

                return
            except Exception as e:
                logging.error(f"[Producer] Connection failed: {e}")
                self.close()
                time.sleep(5)

    def publish(self, message, persistent=True):
        try:
            if not self.channel or self.channel.is_closed:
                logging.info("[Producer] Channel closed, reconnecting...")
                self.connect()

            properties = pika.BasicProperties(
                delivery_mode=2 if persistent else 1  # persistent or transient
            )

            self.channel.basic_publish(
                exchange="",
                routing_key=self.queue,
                body=message,
                properties=properties
            )
            logging.info(f"[Producer] Published message to {self.queue}")

        except (AMQPConnectionError, StreamLostError) as e:
            logging.error(f"[Producer] Connection lost while publishing: {e}")
            self.close()
            time.sleep(2)
            self.publish(message, persistent)  # retry once
        except Exception as e:
            logging.error(f"[Producer] Failed to publish message: {e}", exc_info=True)

    def get_queue_info(self, queue):
        if not self.channel:
            raise RuntimeError("RabbitMQ connection not established. Call connect() first.")

        queue_info= self.channel.queue_declare(queue=queue, passive=True)
        return queue_info

    def close(self):
        """
        Closes the connection to RabbitMQ.
        """
        try:
            if self.channel and self.channel.is_open:
                self.channel.close()
                logging.info("[Consumer] Channel closed")
            if self.connection and self.connection.is_open:
                self.connection.close()
                logging.info("[Consumer] Connection closed")
        except Exception as e:
            logging.error(f"[Consumer] Error while closing: {e}", exc_info=True)
        finally:
            self.channel = None
            self.connection = None