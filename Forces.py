import numpy as np
from GenericConts import G, direc_of_body_to_other_body, dist_of_all_bodies
from Config.ConfigClass import BodyConfig


def newtonian_gravity(r: np.ndarray, mass: np.ndarray) -> np.ndarray:
    dist = dist_of_all_bodies(r)
    unit_vector = direc_of_body_to_other_body(r)
    numerator = G * mass
    denominator = dist ** 2
    """ Dimensions are extended from (N, N) to (N, N, 1) which will result in [[[a], [b], [c]], ...]]] which will be
        multiplied by [[[x, y, z]], ...]]] to result in [[[ax, ay, az], [b.., b.., b..]. ..]]]. This will repeat for
        every row of both uv_coef and unit_vector. The resulting array will be (N, N, 3)"""
    uv_coef = np.divide(numerator, denominator, out=np.zeros_like(numerator * denominator), where=denominator != 0)[
        :, :, np.newaxis]
    result_vec = uv_coef * unit_vector
    return np.sum(result_vec, axis=1)