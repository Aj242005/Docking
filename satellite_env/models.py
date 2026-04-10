import json
from openenv.core.env_server.types import Action, Observation
from pydantic import Field, field_validator
from typing import List

class SatelliteAction(Action):
    """Action for the Satellite environment - controls 6-DOF thrusters."""
    
    translation_thrust: List[float] = Field(
        ..., 
        min_length=3, max_length=3, 
        description="X, Y, Z translational forces. Values bounded [-1.0, 1.0]"
    )
    rotation_torque: List[float] = Field(
        ..., 
        min_length=3, max_length=3, 
        description="Roll, Pitch, Yaw rotational torques. Values bounded [-1.0, 1.0]"
    )
    emergency_brake: bool = Field(
        default=False, 
        description="If True, ignores thrust/torque and fires retro-thrusters to kill all velocity."
    )
    
    @field_validator('translation_thrust', 'rotation_torque', mode='before')
    @classmethod
    def parse_stringified_lists(cls, value):
        if isinstance(value, str):
            try:
                # Converts "[0.1, 0.1, 0.1]" into an actual Python list [0.1, 0.1, 0.1]
                return json.loads(value)
            except json.JSONDecodeError:
                # Fallback neutral vector if the LLM completely scrambles the output
                return [0.0, 0.0, 0.0]
        return value

class SatelliteObservation(Observation):
    """All the observations that an agent can fetch from the environment."""
    relative_position: List[float] = Field(..., description="[x, y, z] distance vector to the docking port")
    relative_rotation: List[float] = Field(..., description="Quaternion [x, y, z, w] representing orientation difference")
    linear_velocity: List[float] = Field(..., description="[vx, vy, vz] movement speed along axes")
    angular_velocity: List[float] = Field(..., description="[wx, wy, wz] rotational spin speed")
    distance_to_target: float = Field(..., description="Absolute Euclidean distance to the target in meters")
    approach_speed: float = Field(..., description="Magnitude of the linear velocity. How fast it is moving overall.")
    alignment_error_degrees: float = Field(..., description="How far off the agent is from perfect rotational alignment (0 means perfectly aligned)")
    fuel_percentage: float = Field(..., description="Remaining thruster fuel. Starts at 100.0.")
    collision_warning: bool = Field(..., description="True if distance is low but approach_speed is dangerously high.")