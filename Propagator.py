import numpy as np
from typing import Callable
from Body import Body

# Finds acceleration from given position. Then averages the slopes at different points inbetween the next time
# step. Finally uses this new average acceleration to update velocity which updates the position.
def rk4(r: np.ndarray, v: np.ndarray, mass: np.ndarray, func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        dt: float):
    k1 = func(r, mass)
    v1 = v + k1 * dt

    k2 = func(r + dt / 2 * v1, mass)
    v2 = v + k2 * dt

    k3 = func(r + dt / 2 * v2, mass)
    v3 = v + k3 * dt

    k4 = func(r + dt * v3, mass)

    v = v + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    r = r + dt * (v1 + 2 * v2 + 2 * v3 + v4) / 6
    return r, v
