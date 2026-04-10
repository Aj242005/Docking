"""
inference.py — Satellite Docking LLM Agent
==========================================
Architecture: Python handles all vector math (thrust direction, braking),
the LLM handles strategy (magnitude scaling only).

KEY FIX: PyBullet applies thrust in the satellite's LOCAL frame (LINK_FRAME).
All world-frame vectors (relative_position, linear_velocity) MUST be rotated
into local frame using the satellite's quaternion before being used as thrust.

Without this transform, braking fires in a random world direction that has
nothing to do with stopping the satellite.
"""

import os
import json
import asyncio
import numpy as np
from openai import OpenAI
from client import SatelliteEnv
from models import SatelliteAction, SatelliteObservation

# ── Environment variables ────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not all([API_BASE_URL, MODEL_NAME, HF_TOKEN]):
    raise ValueError(
        "Missing required environment variables: API_BASE_URL, MODEL_NAME, or HF_TOKEN"
    )

llm_client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# ── Frame transform ───────────────────────────────────────────────────────────

def _world_to_local(world_vec: list, quaternion: list) -> list:
    """
    Rotate a world-frame vector into the body/local frame of the satellite.

    PyBullet quaternion convention: [x, y, z, w]
    To go world → local we apply the INVERSE rotation (conjugate quaternion).

    This is the critical fix: thrust is applied in LOCAL frame by PyBullet,
    so every direction vector must be transformed here first.
    """
    x, y, z, w = quaternion
    # Conjugate = inverse for unit quaternion
    qx, qy, qz, qw = -x, -y, -z, w

    vx, vy, vz = world_vec

    # Rotate v by conjugate quaternion: v' = q * v * q^-1
    # Using optimised formula: v' = v + 2*qw*(q x v) + 2*(q x (q x v))
    qv  = np.array([qx, qy, qz])
    v   = np.array([vx, vy, vz])
    t   = 2.0 * np.cross(qv, v)
    rot = v + qw * t + np.cross(qv, t)
    return rot.tolist()


def _safe_normalize(v: list, fallback: list) -> list:
    arr  = np.array(v, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-6:
        return fallback
    return (arr / norm).tolist()


# ── Phase detection ───────────────────────────────────────────────────────────

def _compute_phase(obs: SatelliteObservation) -> str:
    if obs.collision_warning:
        return "EMERGENCY"
    if obs.approach_speed > 2.0:
        return "BRAKE"
    if obs.distance_to_target > 3.0:
        return "APPROACH"
    if obs.distance_to_target > 0.8:
        return "FINE"
    return "DOCK"


# ── Thrust direction in LOCAL frame ──────────────────────────────────────────

def _compute_base_thrust_local(obs: SatelliteObservation, phase: str) -> list:
    """
    Returns a unit thrust vector already in the satellite's LOCAL frame.
    PyBullet will apply this directly — no further transform needed.

    Strategy per phase:
      BRAKE    → oppose velocity  (world vel → local → negate)
      APPROACH → toward target    (world rel → local → negate)
      FINE     → 70% toward + 30% brake, both in local frame
      DOCK     → tiny toward target in local frame
    """
    quat = obs.relative_rotation   # [x, y, z, w] from PyBullet
    rel  = obs.relative_position   # world frame: agent - parent
    vel  = obs.linear_velocity     # world frame

    if phase == "EMERGENCY":
        return [0.0, 0.0, 0.0]

    if phase == "BRAKE":
        # Oppose velocity in local frame
        vel_local  = _world_to_local(vel, quat)
        return _safe_normalize([-v for v in vel_local], [0.0, 0.0, 0.0])

    if phase == "APPROACH":
        # Move toward target in local frame
        rel_local  = _world_to_local(rel, quat)
        return _safe_normalize([-r for r in rel_local], [0.0, 0.0, 0.0])

    if phase == "FINE":
        rel_local  = _world_to_local(rel, quat)
        vel_local  = _world_to_local(vel, quat)
        toward = np.array(_safe_normalize([-r for r in rel_local], [0.0, 0.0, 0.0]))
        brake  = np.array(_safe_normalize([-v for v in vel_local], [0.0, 0.0, 0.0]))
        blended = 0.7 * toward + 0.3 * brake
        return _safe_normalize(blended.tolist(), [0.0, 0.0, 0.0])

    # DOCK — tiny correction toward target
    rel_local = _world_to_local(rel, quat)
    return _safe_normalize([-r for r in rel_local], [0.0, 0.0, 0.0])


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are the strategy controller for a satellite docking GNC system.
The navigation computer has already computed the CORRECT thrust direction
(already transformed into the satellite's local frame).
Your ONLY job: choose the thrust MAGNITUDE (thrust_scale 0.0 to 1.0).

PHASES and recommended scales:
  BRAKE    — too fast (speed > 2 m/s). Scale: 0.7-1.0. Stop ASAP.
  APPROACH — far away (dist > 3 m).   Scale: 0.3-0.6. Close distance steadily.
  FINE     — close  (0.8-3 m).        Scale: 0.05-0.15. Careful corrections.
  DOCK     — nearly docked (< 0.8 m). Scale: 0.01-0.04. Tiny nudges only.

ADAPTATION RULES:
  delta_reward > 0  → strategy working, hold or slightly increase scale.
  delta_reward < 0  → overshooting or wrong, REDUCE scale.
  reward stalling AND dist not closing → increase scale slightly.

ROTATION: output rotation_torque [3 floats, -1 to 1] to reduce
          alignment_error_degrees toward 0.
          If alignment_error < 10 deg → set all torques to [0, 0, 0].

OUTPUT — ONLY valid JSON, no markdown:
{
  "thrust_scale": <float 0.0 to 1.0>,
  "rotation_torque": [<float>, <float>, <float>],
  "emergency_brake": <bool>
}
""".strip()


# ── Message builder ───────────────────────────────────────────────────────────

def _build_user_message(obs, phase, base_thrust_local, step, last_reward, current_reward):
    delta = current_reward - last_reward

    if step == 0:
        hint = "First step. Choose a thrust_scale appropriate for the phase."
    elif delta > 0.03:
        hint = f"Reward up {delta:+.4f} — working. Maintain or slightly increase scale."
    elif delta < -0.03:
        hint = f"Reward dropped {delta:+.4f} — overshooting. Reduce thrust_scale."
    elif abs(delta) < 0.01 and current_reward < 0.4 and phase == "APPROACH":
        hint = "Stalled. Increase thrust_scale to close distance."
    else:
        hint = f"Reward steady at {current_reward:.4f}. Maintain strategy."

    return (
        f"=== STEP {step:03d} | PHASE: {phase} ===\n"
        f"Reward: {current_reward:.4f}  delta: {delta:+.4f}\n"
        f"Hint: {hint}\n\n"
        f"Telemetry:\n"
        f"  distance_to_target:      {obs.distance_to_target:.3f} m\n"
        f"  approach_speed:          {obs.approach_speed:.3f} m/s\n"
        f"  alignment_error_degrees: {obs.alignment_error_degrees:.1f} deg\n"
        f"  fuel_percentage:         {obs.fuel_percentage:.1f}%\n"
        f"  collision_warning:       {obs.collision_warning}\n\n"
        f"Pre-computed LOCAL-frame thrust direction (ready to apply):\n"
        f"  {[round(v, 4) for v in base_thrust_local]}\n\n"
        f"Choose your thrust_scale, rotation_torque, and emergency_brake."
    )


# ── LLM action ────────────────────────────────────────────────────────────────

def get_llm_action(obs, history, step, last_reward, current_reward):
    phase             = _compute_phase(obs)
    base_thrust_local = _compute_base_thrust_local(obs, phase)

    # Emergency: bypass LLM entirely
    if phase == "EMERGENCY":
        print(f"[STEP {step:03d}] EMERGENCY BRAKE")
        action = SatelliteAction(
            translation_thrust=[0.0, 0.0, 0.0],
            rotation_torque=[0.0, 0.0, 0.0],
            emergency_brake=True,
        )
        note = json.dumps({"thrust_scale": 0.0, "rotation_torque": [0,0,0], "emergency_brake": True})
        history = history + [
            {"role": "user",      "content": f"STEP {step}: EMERGENCY."},
            {"role": "assistant", "content": note},
        ]
        return action, history

    user_msg = _build_user_message(obs, phase, base_thrust_local, step, last_reward, current_reward)
    history  = history + [{"role": "user", "content": user_msg}]

    try:
        response = llm_client.chat.completions.create(
            model=str(MODEL_NAME),
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history[-16:],
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=128,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty LLM response")

        clean = content.strip()
        if clean.startswith("```"):
            clean = "\n".join(l for l in clean.splitlines() if not l.strip().startswith("```"))

        parsed       = json.loads(clean)
        thrust_scale = float(np.clip(parsed.get("thrust_scale", 0.4), 0.0, 1.0))
        torque       = [float(np.clip(v, -1.0, 1.0)) for v in parsed.get("rotation_torque", [0.0, 0.0, 0.0])]
        brake        = bool(parsed.get("emergency_brake", False))

        # Scale the pre-computed LOCAL direction — direction is always correct
        final_thrust = [round(float(np.clip(b * thrust_scale, -1.0, 1.0)), 4)
                        for b in base_thrust_local]

        action = SatelliteAction(
            translation_thrust=final_thrust,
            rotation_torque=torque,
            emergency_brake=brake,
        )

        history = history + [{"role": "assistant", "content": content}]

        print(f"[STEP {step:03d}] Phase={phase:8s} | Scale={thrust_scale:.2f} | "
              f"LocalDir={[round(v,2) for v in base_thrust_local]} | "
              f"Thrust={final_thrust}")
        return action, history

    except Exception as exc:
        print(f"[WARN] LLM error step {step}: {exc}. Physics fallback.")

        # Phase-appropriate default scales
        scale = {"BRAKE": 0.8, "APPROACH": 0.4, "FINE": 0.1, "DOCK": 0.03}.get(phase, 0.3)
        final_thrust = [round(float(np.clip(b * scale, -1.0, 1.0)), 4)
                        for b in base_thrust_local]

        action = SatelliteAction(
            translation_thrust=final_thrust,
            rotation_torque=[0.0, 0.0, 0.0],
            emergency_brake=False,
        )
        note = json.dumps({"thrust_scale": scale, "rotation_torque": [0,0,0], "emergency_brake": False})
        history = history + [{"role": "assistant", "content": note}]
        return action, history


# ── Time budget ───────────────────────────────────────────────────────────────
WALL_CLOCK_LIMIT_S = int(os.getenv("WALL_CLOCK_LIMIT_S", "1080"))  # 18 min hard cutoff
MAX_STEPS          = int(os.getenv("MAX_STEPS", "250"))
TIME_WARN_S        = 60


# ── Main loop ─────────────────────────────────────────────────────────────────

async def main() -> None:
    import time
    episode_start = time.monotonic()

    print("[START] Satellite Docking Agent")
    print(f"        Wall-clock limit : {WALL_CLOCK_LIMIT_S}s ({WALL_CLOCK_LIMIT_S//60}m {WALL_CLOCK_LIMIT_S%60}s)")
    print(f"        Max steps        : {MAX_STEPS}")

    total_reward = 0.0
    step_num     = 0
    last_reward  = 0.0
    history      = []
    stop_reason  = "unknown"

    server_url = os.getenv("SERVER_URL", "ws://localhost:8000")

    async with SatelliteEnv(base_url=server_url) as env:
        result = await env.reset()
        obs    = result.observation
        done   = False

        print(f"[RESET] Spawn dist: {obs.distance_to_target:.2f}m | "
              f"Phase: {_compute_phase(obs)}\n")

        while not done:
            elapsed   = time.monotonic() - episode_start
            remaining = WALL_CLOCK_LIMIT_S - elapsed

            # Hard time cutoff
            if remaining <= 0:
                stop_reason = "timeout"
                print(f"\n[TIMEOUT] Wall-clock limit at step {step_num}. Stopping.")
                break

            # Step cap
            if step_num >= MAX_STEPS:
                stop_reason = "max_steps"
                print(f"\n[STEP CAP] Reached {MAX_STEPS} steps. Stopping.")
                break

            # Warning when close to limit
            if remaining <= TIME_WARN_S and step_num % 5 == 0:
                print(f"[TIME] {remaining:.0f}s remaining | step {step_num}/{MAX_STEPS}")

            dist  = obs.distance_to_target
            speed = obs.approach_speed
            current_reward = float(
                max(0.0, min(0.9, max(0.0, 1.0 - dist / 10.0) - speed * 0.05))
            )

            action, history = get_llm_action(
                obs, history, step_num, last_reward, current_reward
            )

            result      = await env.step(action)
            obs         = result.observation
            dist2       = obs.distance_to_target
            speed2      = obs.approach_speed
            step_reward = float(
                max(0.0, min(0.9, max(0.0, 1.0 - dist2 / 10.0) - speed2 * 0.05))
            )

            total_reward += step_reward
            last_reward   = step_reward
            step_num     += 1
            done          = result.done

            if done:
                stop_reason = "env_done"

            elapsed_now = time.monotonic() - episode_start
            print(
                f"         Reward: {step_reward:.4f} | Dist: {dist2:.3f}m | "
                f"Speed: {speed2:.3f}m/s | Fuel: {obs.fuel_percentage:.1f}% | "
                f"Elapsed: {elapsed_now:.0f}s/{WALL_CLOCK_LIMIT_S}s | Done: {done}"
            )

            await asyncio.sleep(0.05)

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed_total = time.monotonic() - episode_start
    avg_reward    = total_reward / max(step_num, 1)
    final_dist    = obs.distance_to_target

    print(f"\n{'='*60}")
    print(f"[END] Stop reason  : {stop_reason}")
    print(f"      Steps        : {step_num} / {MAX_STEPS}")
    print(f"      Time elapsed : {elapsed_total:.1f}s / {WALL_CLOCK_LIMIT_S}s")
    print(f"      Total reward : {total_reward:.4f}")
    print(f"      Avg/step     : {avg_reward:.4f}")
    print(f"      Final dist   : {final_dist:.3f}m")
    print(f"      Fuel left    : {obs.fuel_percentage:.1f}%")

    if final_dist < 0.5:
        print("      RESULT: DOCKING SUCCESSFUL")
    elif stop_reason == "timeout":
        print("      RESULT: STOPPED — wall-clock limit")
    elif stop_reason == "max_steps":
        print(f"      RESULT: STOPPED — {MAX_STEPS} step cap")
    elif obs.fuel_percentage <= 0:
        print("      RESULT: FAILED — fuel exhausted")
    elif final_dist > 15.0:
        print("      RESULT: FAILED — drifted out of range")
    else:
        print(f"      RESULT: Ended at {final_dist:.2f}m from target")


if __name__ == "__main__":
    asyncio.run(main())