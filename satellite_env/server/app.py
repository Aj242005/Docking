# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Satellite Env Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
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
    """Entry point for uv run and multi-mode deployment."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()