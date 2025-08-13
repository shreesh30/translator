from src.orchestration_service.orchestration_manager import OrchestrationManager
from src.utils.utils import Utils

def run_orchestration_manager():
    orchestration_manager = OrchestrationManager(spot_fleet_config_path="config/spot_fleet_request.json")
    orchestration_manager.run()


if __name__ == "__main__":
    Utils.setup_logging("orchestration_service.log")

    run_orchestration_manager()