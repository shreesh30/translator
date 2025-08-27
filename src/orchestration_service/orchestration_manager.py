import json
import logging
import os
import time
from enum import Enum

import boto3

from src.service.rabbitmq_producer import RabbitMQProducer
from src.utils.utils import Utils

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class FleetStatus(Enum):
    ACTIVE = "active"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PENDING = "pending"
    UNKNOWN = "unknown"
    ERROR = "error"

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
        """
        Check the Spot Fleet request status.
        Returns a FleetStatus enum:
          - FleetStatus.ACTIVE
          - FleetStatus.FAILED
          - FleetStatus.CANCELLED
          - FleetStatus.PENDING
          - FleetStatus.UNKNOWN (unexpected state)
          - FleetStatus.ERROR (API error)
        """
        try:
            response = self.ec2_client.describe_spot_fleet_requests(
                SpotFleetRequestIds=[fleet_id]
            )
            configs = response.get("SpotFleetRequestConfigs", [])
            if not configs:
                logger.warning(f"No configuration returned for fleet {fleet_id}.")
                return FleetStatus.UNKNOWN

            state = configs[0].get("SpotFleetRequestState", "unknown")
            logger.info(f"Fleet {fleet_id} status: {state}")

            if state == "active":
                return FleetStatus.ACTIVE
            elif state == "failed":
                return FleetStatus.FAILED
            elif state == "cancelled":
                return FleetStatus.CANCELLED
            elif state in ("submitted", "pending_fulfillment", "modifying"):
                return FleetStatus.PENDING
            else:
                logger.warning(f"Unexpected Spot Fleet state '{state}' for fleet {fleet_id}.")
                return FleetStatus.UNKNOWN

        except self.ec2_client.exceptions.ClientError as e:
            logger.error(f"Failed to describe fleet {fleet_id}: {e}")
            return FleetStatus.ERROR

    def run(self, retry_interval=30):
        """Continuously try to request a Spot Fleet until it becomes active."""
        fleet_id = None

        while True:
            # Request a new fleet if we don't have one
            if fleet_id is None:
                fleet_id = self.request_spot_fleet()
                if fleet_id is None:
                    logger.info(f"Spot Fleet request failed. Retrying in {retry_interval}s...")
                    time.sleep(retry_interval)
                    continue

            # Check the fleet's status
            status = self.check_fleet_status(fleet_id)

            if status == FleetStatus.ACTIVE:
                logger.info(f"Spot Fleet {fleet_id} is active.")
                break  # Exit the loop successfully
            elif status in (FleetStatus.FAILED, FleetStatus.CANCELLED, FleetStatus.ERROR):
                logger.warning(f"Spot Fleet {fleet_id} {status.value}. Sending a new request...")
                fleet_id = None  # Reset to try again
                time.sleep(retry_interval)
            elif status in (FleetStatus.PENDING, FleetStatus.UNKNOWN):
                logger.info(f"Fleet {fleet_id} in {status} state. Checking again in {retry_interval}s...")
                time.sleep(retry_interval)

