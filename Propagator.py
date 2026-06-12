import numpy as np
from typing import Callable
from Body import Body

# Finds acceleration from given position. Then averages the slopes at different points inbetween the next time
# step. Finally uses this new average acceleration to update velocity which updates the position.
def rk4(r: np.ndarray, v: np.ndarray, mass: np.ndarray, func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        dt: float):
    k1 = func(r, mass)
    v1 = v

    k2 = func(r + dt / 2 * v1, mass)
    v2 = v + k1 * dt / 2

    k3 = func(r + dt / 2 * v2, mass)
    v3 = v + k2 * dt / 2

    k4 = func(r + dt * v3, mass)
    v4 = v + k3 * dt

    v = v + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    r = r + dt * (v1 + 2 * v2 + 2 * v3 + v4) / 6
    return r, v

def leapfrog(r: np.ndarray, v: np.ndarray, mass: np.ndarray, func: Callable[[np.ndarray, np.ndarray], np.ndarray],
        dt: float):
    a = func(r, mass)
    x_new = r + v * dt + 0.5 * a * dt ** 2
    v_new = v + 0.5 * (a + func(x_new, mass)) * dt

    return x_new, v_new
