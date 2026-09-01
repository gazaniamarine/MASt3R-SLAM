"""Placeholders for dependencies the HM3D planning path does not need.

The maze2d pipeline pulls mujoco_py, gym, d4rl and qpth in at *import* time:
diffuser/models/diffusion.py imports diffuser.utils, which imports rendering,
which imports mujoco_py. The HM3D path replaces the environment (the occupancy
grid is the world) and the renderer, and never calls the QP-based ablations, so
none of those have to be installed to plan on a BEV map.

Importing them normally is still the default -- when the package is present it
comes back untouched, and the maze2d path behaves exactly as before. Only when
one is missing do we substitute a proxy that imports cleanly and raises at the
point of first use, so the failure reads "mujoco_py is required for MuJoCo
offscreen rendering" instead of an import error three modules away.
"""
import importlib


class MissingDependency:

    def __init__(self, name, needed_for):
        self._name = name
        self._needed_for = needed_for

    def _fail(self):
        raise ImportError(
            f"'{self._name}' is not installed; it is required for {self._needed_for}. "
            "The HM3D planning path does not use it -- install it only if you need "
            "the original maze2d/d4rl pipeline."
        )

    def __getattr__(self, attr):
        self._fail()

    def __call__(self, *args, **kwargs):
        self._fail()


def optional_import(name, needed_for):
    try:
        return importlib.import_module(name)
    except ImportError:
        return MissingDependency(name, needed_for)
