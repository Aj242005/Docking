import os
import json
import asyncio
from openai import OpenAI
from client import SatelliteEnv
from models import SatelliteAction

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

if not all([API_BASE_URL, MODEL_NAME, HF_TOKEN]):
    raise ValueError("Missing required environment variables: API_BASE_URL, MODEL_NAME, or HF_TOKEN")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = """
You are an AI piloting a 6-DOF satellite in zero gravity.
Your goal is to dock with the parent satellite (Distance = 0, Speed = 0).
Because there is no friction, if you have a linear_velocity, you will keep drifting unless you fire counter-thrusters to stop!

Rules:
1. If distance_to_target is large, use translation_thrust to move closer.
2. If approach_speed is high, you MUST fire opposite thrusters to slow down.
3. If collision_warning is true, set emergency_brake to true.

Output strictly valid JSON exactly matching this format:
{
    "translation_thrust": [x, y, z],
    "rotation_torque": [pitch, yaw, roll],
    "emergency_brake": false
}
YOU MUST NEVER USE A THRUST VALUE GREATER THAN 0.1 . IF YOU ARE MOVING FASTER THAN 0.5 M/S, YOU MUST FIRE THRUSTERS IN THE OPPOSITE DIRECTION TO SLOW DOWN (this is a strict command for implementation).
Note: Arrays must contain exactly 3 floats between -3.0(generally for the reverse thrust) and 2.0 and try to experiment with the speed of the aircraft so as to get the optimimum speed and you reach to the primary satellite in the minimum and fastest time possible.
"""

def get_llm_action(obs) -> SatelliteAction:
    obs_json = obs.model_dump_json(indent=2)
    
    try:
        response = client.chat.completions.create(
            model=str(MODEL_NAME),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Current State:\n{obs_json}"}
            ],
            temperature=1,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("LLM returned empty content")
        action_dict = json.loads(content)
        return SatelliteAction(**action_dict)
    except Exception as e:
        print(f"LLM Error/Hallucination: {e}. Defaulting to zero thrust.")
        # Fallback to prevent crash if LLM outputs bad JSON
        return SatelliteAction(
            translation_thrust=[0.0, 0.0, 0.0],
            rotation_torque=[0.0, 0.0, 0.0],
            emergency_brake=False
        )

async def main():
    print("[START] Initializing Episode 1")
    
    total_reward = 0.0
    step_num = 0
    
    async with SatelliteEnv(base_url="ws://localhost:8000") as env:
        result = await env.reset()
        obs = result.observation
        done = False
        
        while not done:
            action = get_llm_action(obs)
            
            result = await env.step(action)
            dist = result.observation.distance_to_target
            speed = result.observation.approach_speed
        
            distance_score = max(0, 1.0 - (dist / 10.0))
            speed_penalty = speed * 0.05
            client_reward = float(max(0.0, min(0.9, distance_score - speed_penalty)))
        
            print(f"[STEP] Action: Thrust {action.translation_thrust} | Brake: {action.emergency_brake} | Reward: {client_reward:.4f} | Done: {result.done}")
            print(f"       Telemetry: Dist: {dist:.2f}m | Speed: {speed:.2f}m/s\n")
        
            await asyncio.sleep(0.2)
            done = result.done

    print(f"[END] Episode Finished. Total Steps: {step_num} | Final Total Reward: {total_reward:.4f*10000}")

if __name__ == "__main__":
    asyncio.run(main())