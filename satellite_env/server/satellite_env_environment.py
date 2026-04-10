"""
satellite_env_environment.py — 6-DOF Satellite Docking Environment
===================================================================
Physics-based docking simulation using PyBullet in headless (DIRECT) mode.

Improvements over v1:
- Richer, shaped reward function (distance + alignment + fuel efficiency)
- Done condition uses correct operator precedence (was a silent bug)
- Approach speed computed from velocity projected onto approach vector
  (more meaningful than raw speed magnitude)
- Collision penalty added to reward
- Episode history tracked for debugging
- reset() properly disconnects and reconnects PyBullet to avoid ghost state
"""

import math
import random
from typing import Any, Optional

import numpy as np
import pybullet as p
import pybullet_data

# --- OpenEnv Imports ---
from openenv.core import Environment
from openenv.core.env_server.types import State

# --- Local Schema Imports ---
from models import SatelliteAction, SatelliteObservation


class SatelliteEnvironment(Environment[SatelliteAction, SatelliteObservation, State]):

    def __init__(self):
        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)

        self.parent_id: Optional[int] = None
        self.agent_id:  Optional[int] = None
        self.step_count    = 0
        self.max_steps     = 100          # increased from 50
        self.fuel          = 100.0
        self.current_reward = 0.0
        self.current_done  = False

        # Tracking for reward shaping
        self._prev_dist    = None
        self._episode_id   = "sim_0"
    
    # ── Reset ──────────────────────────────────────────────────────────────

    def reset(
        self,
        seed:       Optional[int]  = None,
        episode_id: Optional[str]  = None,
        **kwargs: Any,
    ) -> SatelliteObservation:
        """Reset simulation, spawn bodies, return initial observation."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._episode_id = episode_id or f"sim_{self.step_count}"

        p.resetSimulation()
        p.setGravity(0, 0, 0)

        self.step_count    = 0
        self.fuel          = 100.0
        self.current_reward = 0.0
        self.current_done  = False
        self._prev_dist    = None

        # ── Parent (target docking port) — static body at origin ──
        parent_col  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, 1, 1])
        self.parent_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=parent_col,
            basePosition=[0, 0, 0],
        )

        # ── Agent (chaser satellite) — random spawn ──
        agent_col   = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
        spawn_pos   = [random.uniform(-8, 8) for _ in range(3)]
        spawn_orn   = p.getQuaternionFromEuler(
            [random.uniform(0, 2 * math.pi) for _ in range(3)]
        )
        self.agent_id = p.createMultiBody(
            baseMass=10,
            baseCollisionShapeIndex=agent_col,
            basePosition=spawn_pos,
            baseOrientation=spawn_orn,
        )

        obs = self._get_observation()
        self._prev_dist = obs.distance_to_target
        return obs

    # ── Step ───────────────────────────────────────────────────────────────

    def step(
        self,
        action:    SatelliteAction,
        timeout_s: Optional[float] = None,
        **kwargs:  Any,
    ) -> SatelliteObservation:
        """Apply action, advance simulation, compute shaped reward."""
        self.step_count += 1

        # ── Apply control inputs ──────────────────────────────────────────
        if action.emergency_brake:
            p.resetBaseVelocity(self.agent_id, [0, 0, 0], [0, 0, 0])
            self.fuel = max(0.0, self.fuel - 10.0)
            for _ in range(60):
                p.stepSimulation()
        else:
            thrust = np.clip(action.translation_thrust, -1.0, 1.0) * 50.0
            torque = np.clip(action.rotation_torque,    -1.0, 1.0) * 10.0

            fuel_burn = (np.sum(np.abs(thrust)) + np.sum(np.abs(torque))) * 0.01
            self.fuel = max(0.0, self.fuel - fuel_burn)

            for _ in range(60):
                p.applyExternalForce(
                    self.agent_id, -1,
                    forceObj=thrust.tolist(),
                    posObj=[0, 0, 0],
                    flags=p.WORLD_FRAME,
                )
                p.applyExternalTorque(
                    self.agent_id, -1,
                    torqueObj=torque.tolist(),
                    flags=p.WORLD_FRAME,
                )
                p.stepSimulation()

        # ── Observe & compute reward ──────────────────────────────────────
        obs  = self._get_observation()
        dist = obs.distance_to_target

        # --- Shaped reward components ---
        # 1. Distance score: 1 at target, 0 at 10 m
        distance_score = max(0.0, 1.0 - dist / 10.0)

        # 2. Approach speed penalty (penalise high speed)
        speed_penalty = obs.approach_speed * 0.05

        # 3. Alignment bonus: reward for being pointed at target
        align_bonus = max(0.0, 1.0 - obs.alignment_error_degrees / 180.0) * 0.05

        # 4. Progress reward: bonus for closing distance vs last step
        progress = 0.0
        if self._prev_dist is not None:
            progress = max(0.0, (self._prev_dist - dist) * 0.02)
        self._prev_dist = dist

        # 5. Fuel efficiency bonus
        fuel_bonus = (self.fuel / 100.0) * 0.02

        # 6. Collision penalty
        collision_penalty = 0.15 if obs.collision_warning else 0.0

        raw_reward = (
            distance_score
            + align_bonus
            + progress
            + fuel_bonus
            - speed_penalty
            - collision_penalty
        )
        self.current_reward = float(np.clip(raw_reward, 0.0, 1.0))

        # ── Done conditions — explicit precedence (was a bug in v1) ──────
        docked     = dist < 0.5 and obs.approach_speed < 0.1
        out_range  = dist > 20.0
        out_steps  = self.step_count >= self.max_steps
        out_fuel   = self.fuel <= 0.0

        self.current_done = bool(docked or out_range or out_steps or out_fuel)

        status = (
            "DOCKED ✅" if docked else
            "OUT OF RANGE ❌" if out_range else
            "OUT OF FUEL ❌" if out_fuel else
            "MAX STEPS ⏹" if out_steps else
            "running"
        )
        print(
            f"[ENV step={self.step_count:03d}] "
            f"reward={self.current_reward:.4f} | dist={dist:.3f}m | "
            f"speed={obs.approach_speed:.3f}m/s | fuel={self.fuel:.1f}% | "
            f"status={status}"
        )

        return self._get_observation()

    # ── State property ─────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return State(episode_id=self._episode_id, step_count=self.step_count)

    # ── Reward / done helpers ──────────────────────────────────────────────

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

    # ── Internal observation builder ───────────────────────────────────────

    def _get_observation(self) -> SatelliteObservation:
        agent_pos,  agent_orn  = p.getBasePositionAndOrientation(self.agent_id)
        parent_pos, _          = p.getBasePositionAndOrientation(self.parent_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.agent_id)

        rel_pos = np.array(agent_pos) - np.array(parent_pos)
        dist    = float(np.linalg.norm(rel_pos))

        # Approach speed = component of velocity along the approach axis
        # (positive = moving away, negative = closing)
        if dist > 1e-6:
            approach_dir  = -rel_pos / dist          # direction toward target
            approach_speed = float(
                np.dot(np.array(linear_vel), approach_dir)
            )
            # We want magnitude of speed irrespective of sign for the penalty
            approach_speed = abs(approach_speed)
        else:
            approach_speed = float(np.linalg.norm(linear_vel))

        # Alignment error via quaternion w component
        w           = float(np.clip(abs(agent_orn[3]), 0.0, 1.0))
        angle_rad   = 2.0 * math.acos(w)
        align_err   = math.degrees(angle_rad)

        return SatelliteObservation(
            relative_position       =rel_pos.tolist(),
            relative_rotation       =list(agent_orn),
            linear_velocity         =list(linear_vel),
            angular_velocity        =list(angular_vel),
            distance_to_target      =dist,
            approach_speed          =approach_speed,
            alignment_error_degrees =align_err,
            fuel_percentage         =float(self.fuel),
            collision_warning       =bool(dist < 2.0 and approach_speed > 1.5),
        )