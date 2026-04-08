import pybullet as p
import pybullet_data
import numpy as np
import random
import math
from typing import Optional, Any

# --- OpenEnv Imports ---
from openenv.core import Environment
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

# --- Local Schema Imports ---
from models import SatelliteAction, SatelliteObservation

class SatelliteEnvironment(Environment[SatelliteAction, SatelliteObservation, State]):
    def __init__(self):
        # Initialize PyBullet in headless mode
        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)

        self.parent_id = None
        self.agent_id = None
        self.step_count = 0
        self.max_steps = 50
        self.fuel = 100.0
        self.current_reward = 0.0
        self.current_done = False

    def reset(
        self, 
        seed: Optional[int] = None, 
        episode_id: Optional[str] = None, 
        **kwargs: Any
    ) -> SatelliteObservation:
        """
        Resets the environment to an initial state and returns the initial observation.
        Strictly compliant with OpenEnv's base Environment signature.
        """
        # Handle the random seed if the grader provides one
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        p.resetSimulation()
        p.setGravity(0, 0, 0)
        self.step_count = 0
        self.fuel = 100.0

        # Create Parent (Target)
        parent_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, 1])
        self.parent_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=parent_col, basePosition=[0, 0, 0])

        # Create Agent
        agent_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        spawn_pos = [random.uniform(-5, 5) for _ in range(3)]
        spawn_orn = p.getQuaternionFromEuler([random.uniform(0, 2*math.pi) for _ in range(3)])
        self.agent_id = p.createMultiBody(baseMass=10, baseCollisionShapeIndex=agent_col, 
                                          basePosition=spawn_pos, baseOrientation=spawn_orn)

        return self._get_observation()

    def step(
        self, 
        action: SatelliteAction, 
        timeout_s: Optional[float] = None, 
        **kwargs: Any
    ) -> SatelliteObservation:
        """
        Applies the action, steps the physics simulation, and calculates the reward.
        Strictly compliant with OpenEnv's base Environment signature.
        """
        self.step_count += 1

        # 1. Apply Actions
        if action.emergency_brake:
            p.resetBaseVelocity(self.agent_id, [0, 0, 0], [0, 0, 0])
            self.fuel = max(0.0, self.fuel - 10.0)
            
            # Step the engine forward with NO force
            for _ in range(60):
                p.stepSimulation()
        else:
            thrust = np.clip(action.translation_thrust, -1.0, 1.0) * 50.0
            torque = np.clip(action.rotation_torque, -1.0, 1.0) * 10.0
            
            fuel_burn = (np.sum(np.abs(thrust)) + np.sum(np.abs(torque))) * 0.01
            self.fuel = max(0.0, self.fuel - fuel_burn)

            # THE FIX: Apply the force inside the loop so it burns continuously!
            for _ in range(60):
                p.applyExternalForce(self.agent_id, -1, forceObj=thrust.tolist(), posObj=[0,0,0], flags=p.LINK_FRAME)
                p.applyExternalTorque(self.agent_id, -1, torqueObj=torque.tolist(), flags=p.LINK_FRAME)
                p.stepSimulation()

        # 2. Get State and Calculate Reward
        obs = self._get_observation()
        dist = obs.distance_to_target
        speed = obs.approach_speed
        align_err = obs.alignment_error_degrees

        distance_score = max(0, 1.0 - (dist / 10.0))
        speed_penalty = speed * 0.05
        
        self.current_reward = float(np.clip(distance_score - speed_penalty, 0.0, 0.9))
        self.current_done = bool(
            dist < 0.5 and speed < 0.1 or 
            dist > 15.0 or 
            self.step_count >= self.max_steps or 
            self.fuel <= 0
        )
        print(f"SERVER CALCULATED REWARD: {self.current_reward:.4f} | DONE: {self.current_done}")
        # OpenEnv expects just the Observation back!
        return self._get_observation()
    
    @property
    def state(self) -> State:
        """
        Returns the current internal state/metadata of the environment.
        Uses @property to strictly comply with OpenEnv's base Environment signature.
        """
        return State(episode_id="sim_1", step_count=self.step_count)

    def _get_observation(self) -> SatelliteObservation:
        """Helper method to extract PyBullet data and format it into our Pydantic model."""
        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.agent_id)
        parent_pos, parent_orn = p.getBasePositionAndOrientation(self.parent_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.agent_id)

        rel_pos = np.array(agent_pos) - np.array(parent_pos)
        dist = np.linalg.norm(rel_pos)
        speed = np.linalg.norm(linear_vel)

        dot_product = abs(agent_orn[3])
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle_rad = 2 * math.acos(dot_product)
        align_err = math.degrees(angle_rad)

        return SatelliteObservation(
            relative_position=rel_pos.tolist(),
            relative_rotation=list(agent_orn),
            linear_velocity=list(linear_vel),
            angular_velocity=list(angular_vel),
            distance_to_target=float(dist),
            approach_speed=float(speed),
            alignment_error_degrees=float(align_err),
            fuel_percentage=float(self.fuel),
            collision_warning=bool(dist < 2.0 and speed > 1.5)
        )
    @property
    def reward(self) -> float:
        return self.current_reward

    def get_reward(self) -> float:
        return self.current_reward

    @property
    def done(self) -> bool:
        return self.current_done

    def is_done(self) -> bool:
        return self.current_done

# Instantiate the environment for the web server to bind to
env = SatelliteEnvironment()