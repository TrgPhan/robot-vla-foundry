"""Process-local rendering settings for simulation launched from notebook kernels."""

import os
import sys


def simulation_environment(parent=None):
    env = dict(os.environ if parent is None else parent)
    # Notebook kernels export an inline backend unavailable in the isolated sim environment.
    env["MPLBACKEND"] = "Agg"
    if sys.platform == "linux":
        renderer = env.setdefault("MUJOCO_GL", "egl")
        if renderer in ["egl", "osmesa"]:
            env.setdefault("PYOPENGL_PLATFORM", renderer)
    return env
