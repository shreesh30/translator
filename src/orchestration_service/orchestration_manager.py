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

    def is_spot_fleet_running(self):
        response = self.ec2_client.describe_spot_fleet_instances()
        logger.info(f'Spot Fleet Response: {response}')
        for req in response.get("SpotFleetRequestConfigs", []):
            state = req["SpotFleetRequestState"]
            if state in ("active", "submitted", "modifying"):  # still alive
                logger.info(f"Existing spot fleet found: {req['SpotFleetRequestId']} (state={state})")
                return True
        return False

    def request_spot_fleet(self):
        logger.info("Requesting spot fleet...")
        config_path = os.path.join(BASE_DIR, self.spot_fleet_config_path)

        with open(config_path, "r") as f:
            config = json.load(f)

        try:
            response = self.ec2_client.request_spot_fleet(
                SpotFleetRequestConfig=config
            )
            fleet_id = response["SpotFleetRequestId"]
            logger.info(f"Spot Fleet Requested: {fleet_id}")
            return fleet_id
        except self.ec2_client.exceptions.ClientError as e:
            logger.error(f"Failed to request spot fleet: {e}")
            return None

    def check_fleet_status(self, fleet_id):
        """Check the Spot Fleet request status.
        Returns True if active, False if failed/cancelled, None if pending.
        """
        try:
            response = self.ec2_client.describe_spot_fleet_requests(SpotFleetRequestIds=[fleet_id])
            state = response['SpotFleetRequestConfigs'][0]['SpotFleetRequestState']
            logger.info(f"Fleet {fleet_id} status: {state}")

            if state == "active":
                return True
            elif state in ["failed", "cancelled"]:
                return False
            else:  # submitted, pending_fulfillment
                return None

        except self.ec2_client.exceptions.ClientError as e:
            logger.error(f"Failed to describe fleet {fleet_id}: {e}")
            return False

    def run(self, retry_interval=30):
        """Continuously try to request a Spot Fleet until it becomes active."""
        fleet_id = None
        fleet_active = False

        while not fleet_active:
            # Step 1: Request a new fleet if we don't have one
            if fleet_id is None:
                fleet_id = self.request_spot_fleet()
                if fleet_id is None:
                    logger.info(f"Request failed. Retrying in {retry_interval}s...")
                    time.sleep(retry_interval)
                    continue

            # Step 2: Check the fleet's status
            status = self.check_fleet_status(fleet_id)

            if status is True:
                logger.info(f"Spot Fleet {fleet_id} is active")
                fleet_active = True
            elif status is False:
                logger.warning(f"Spot Fleet {fleet_id} failed. Sending a new request...")
                fleet_id = None  # Reset to try again in the next iteration
                time.sleep(retry_interval)
            else:  # Status is pending
                logger.info(f"Fleet {fleet_id} pending. Waiting {retry_interval}s before checking again...")
                time.sleep(retry_interval)


