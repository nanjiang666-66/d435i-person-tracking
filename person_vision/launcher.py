"""Launch the tracker with the activated virtual environment, if any.

ROS 2/colcon may generate a console script bound to the system Python even when
a virtual environment was active during the build. Keep heavy ML imports out of
this module so the launcher can hand off to the intended interpreter first.
"""

import os
import sys
from pathlib import Path


def main():
    venv = os.environ.get("VIRTUAL_ENV")
    if venv and sys.prefix == sys.base_prefix:
        python = Path(venv) / "bin" / "python3"
        if not python.is_file():
            python = Path(venv) / "bin" / "python"
        if not python.is_file():
            raise SystemExit(f"Virtual environment Python not found in {venv}")
        os.execv(str(python), [str(python), "-m", "person_vision.tracker_node", *sys.argv[1:]])

    from person_vision.tracker_node import main as run_tracker

    return run_tracker()
