import json
import logging
import time

import boto3
import pika

from src.utils.utils import Utils

logger = logging.getLogger(__name__)

class OrchestrationManager:
    def __init__(self, spot_fleet_config_path):
        self.spot_fleet_config_path = spot_fleet_config_path
        self.ec2_client = boto3.client("ec2", region_name="us-east-1")

    @staticmethod
    def get_queue_message_count(queue):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host= Utils.KEY_LOCALHOST))
        channel = connection.channel()
        queue_info = channel.queue_declare(queue=queue, durable=True)
        message_count = queue_info.method.message_count
        connection.close()
        return message_count

    def is_spot_fleet_running(self, fleet_id):
        response = self.ec2_client.describe_spot_fleet_instances(SpotFleetRequestId=fleet_id)
        return len(response.get("ActiveInstances", [])) > 0

    def request_spot_fleet(self):
        logger.info("Requesting spot fleet...")
        with open(self.spot_fleet_config_path, "r") as f:
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
            tasks_count = self.get_queue_message_count(Utils.QUEUE_TASKS)
            logger.info(f"Total Tasks: {tasks_count}")

            if tasks_count > 0:
                if not fleet_id or not self.is_spot_fleet_running(fleet_id):
                    fleet_id = self.request_spot_fleet()
            else:
                logger.info("No tasks in queue. Waiting...")

            time.sleep(10)  # check every 10 seconds


