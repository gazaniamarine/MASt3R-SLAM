try:
    from .registration import register_environments
    registered_environments = register_environments()
except ImportError:
    # These are the gym locomotion envs (Hopper/HalfCheetah/Walker2d/Ant) and
    # they need gym at import time. Neither maze2d nor the HM3D planning path
    # touches them, so a missing gym should not block importing `diffuser`.
    registered_environments = ()
