# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Satellite Env Environment.

Endpoints:
    POST /reset   — Reset the environment
    POST /step    — Execute an action
    GET  /state   — Get current environment state
    GET  /schema  — Get action/observation schemas
    WS   /ws      — WebSocket for persistent sessions

Run (dev):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

Run (prod):
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: uv sync"
    ) from e

try:
    from models import SatelliteAction, SatelliteObservation
    from server.satellite_env_environment import SatelliteEnvironment
except ModuleNotFoundError:
    from models import SatelliteAction, SatelliteObservation
    from server.satellite_env_environment import SatelliteEnvironment

# Create the FastAPI app
app = create_app(
    SatelliteEnvironment,
    SatelliteAction,
    SatelliteObservation,
    env_name="satellite_env",
    max_concurrent_envs=1,
)


def main() -> None:
    """
    Zero-argument entry point required by the OpenEnv multi-mode deployment
    validator and by the [project.scripts] uv entry point.

    Called by:
        uv run --project . server
        python -m server.app
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()