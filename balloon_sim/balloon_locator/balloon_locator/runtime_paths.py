"""Repository-relative external build locations, with explicit local overrides."""
import os
from pathlib import Path


def runtime_paths(workspace=None, environ=None):
    env = os.environ if environ is None else environ
    workspace = Path(workspace or env.get('BALLOON_WORKSPACE_ROOT', Path.cwd())).resolve()
    px4 = Path(env.get('BALLOON_PX4_ROOT', workspace/'external/PX4-Autopilot')).expanduser().resolve()
    return {
        'px4_root': px4,
        'px4_binary': Path(env.get('BALLOON_PX4_BINARY', px4/'build/px4_sitl_default/bin/px4')).expanduser().resolve(),
        'px4_plugin_dir': Path(env.get('BALLOON_PX4_PLUGIN_DIR', px4/'build/balloon_gazebo_classic')).expanduser().resolve(),
    }
