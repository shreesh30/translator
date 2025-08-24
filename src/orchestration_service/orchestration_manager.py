import json
import logging
import os
import time

import boto3

from src.service.rabbitmq_producer import RabbitMQProducer
from src.utils.utils import Utils

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class OrchestrationManager:
    def __init__(self, spot_fleet_config_path):
        self.spot_fleet_config_path = spot_fleet_config_path
        self.ec2_client = boto3.client("ec2", region_name="us-east-1")

    @staticmethod
    def get_task_count():
        producer = RabbitMQProducer(host=Utils.KEY_LOCALHOST, queue=Utils.QUEUE_TASKS)
        producer.connect()
        queue_info = producer.get_queue_info(Utils.QUEUE_TASKS)
        message_count = queue_info.method.message_count
        producer.close()

        return message_count

    def is_spot_fleet_running(self, fleet_id):
        response = self.ec2_client.describe_spot_fleet_instances(SpotFleetRequestId=fleet_id)
        return len(response.get("ActiveInstances", [])) > 0

    def request_spot_fleet(self):
        logger.info("Requesting spot fleet...")
        config_path = os.path.join(BASE_DIR, self.spot_fleet_config_path)

        with open(config_path, "r") as f:
            config = json.load(f)
        response = self.ec2_client.request_spot_fleet(
            SpotFleetRequestConfig=config
        )
        fleet_id = response["SpotFleetRequestId"]
        logger.info(f"Spot Fleet Requested: {fleet_id}")
        return fleet_id

    def run(self):
        logger.info("Starting OrchestrationManager...")
        fleet_id = None

        while True:
            tasks_count = self.get_task_count()
            logger.info(f"Total Tasks: {tasks_count}")

            if tasks_count > 0:
                if not fleet_id or not self.is_spot_fleet_running(fleet_id):
                    fleet_id = self.request_spot_fleet()
            else:
                logger.info("No tasks in queue. Waiting...")

            time.sleep(30)  # check every 30 seconds


