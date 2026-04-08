from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import SatelliteAction, SatelliteObservation

class SatelliteEnv(
    EnvClient[SatelliteAction, SatelliteObservation, State]
):
    """
    Client for the 6-DOF Satellite Docking Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    """

    def _step_payload(self, action: SatelliteAction) -> Dict:
        """
        Convert SatelliteAction to JSON payload for step message.
        """
        # We extract the 3 variables from our Action model to send to the server
        return {
            "translation_thrust": action.translation_thrust,
            "rotation_torque": action.rotation_torque,
            "emergency_brake": action.emergency_brake,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SatelliteObservation]:
        """
        Parse server response into StepResult[SatelliteObservation].
        """
        obs_data = payload.get("observation", {})
        
        # We safely extract our 9 variables from the server's JSON response
        observation = SatelliteObservation(
            relative_position=obs_data.get("relative_position", [0.0, 0.0, 0.0]),
            relative_rotation=obs_data.get("relative_rotation", [0.0, 0.0, 0.0, 1.0]),
            linear_velocity=obs_data.get("linear_velocity", [0.0, 0.0, 0.0]),
            angular_velocity=obs_data.get("angular_velocity", [0.0, 0.0, 0.0]),
            distance_to_target=obs_data.get("distance_to_target", 0.0),
            approach_speed=obs_data.get("approach_speed", 0.0),
            alignment_error_degrees=obs_data.get("alignment_error_degrees", 0.0),
            fuel_percentage=obs_data.get("fuel_percentage", 100.0),
            collision_warning=obs_data.get("collision_warning", False),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward") if payload.get("reward") is not None else 0.0,
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )