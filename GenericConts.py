from Body import Body
import numpy as np

G = 6.674 * 10 ** -11
EARTH_MASS = 5.9722e24

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