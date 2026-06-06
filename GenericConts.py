import yaml
import numpy as np
from pathlib import Path



G = 6.674 * 10 ** -11 #  m^3 * kg^−1 * s^−2
EARTH_MASS = 5.9722e24 # kg
EARTH_ACCEL = -9.81 # m / s^2

# This returns a unit vector representing the direction of the main_body to the other body
def direc_of_body_to_other_body(r: np.ndarray) -> np.ndarray:
    r2 = r[:, np.newaxis]
    numerator = r - r2
    denominator = np.linalg.norm(numerator, axis = -1, keepdims = True)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator * denominator), where=denominator != 0)


def dist_of_all_bodies(r: np.ndarray) -> np.ndarray:
    r_broadcast = r[:, np.newaxis]
    dist_vec = r - r_broadcast
    return np.linalg.norm(dist_vec, axis = 2)

def load_yaml(name: str) -> dict:
    directory_name = Path(__file__).resolve().parent / "Config" / "InitialConsts" / name
    with open(directory_name, 'r') as f:
        return yaml.safe_load(f)