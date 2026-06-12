import numpy as np
from GenericConts import G


def two_body_potential_gradient(r: np.ndarray, b1_pos: np.ndarray, b2_pos: np.ndarray,
                                m1: float, m2: float, omega: float) -> np.ndarray:
    body1_deriv = G * 1 / 2 * m1 * ((r[0] - b1_pos[0]) ** 2 + (r[1] - b1_pos[1]) ** 2 + (r[2] - b1_pos[2]) ** 2) ** (-3 / 2)
    body2_deriv = G * 1 / 2 * m2 * ((r[0] - b2_pos[0]) ** 2 + (r[1] - b2_pos[1]) ** 2 + (r[2] - b2_pos[2]) ** 2) ** (-3 / 2)
    du_dx = -body1_deriv * 2 * (r[0] - b1_pos[0]) - body2_deriv * 2 * (r[0] - b2_pos[0])  - omega ** 2 * r[0]
    du_dy = -body1_deriv * 2 * (r[1] - b1_pos[1]) - body2_deriv * 2 * (r[1] - b2_pos[1]) - omega ** 2 * r[1]
    du_dz = -body1_deriv * 2 * (r[2] - b1_pos[2]) - body2_deriv * 2 * (r[2] - b2_pos[2])

    return np.array([du_dx, du_dy, du_dz])


def angular_velocity(r: float, m1: float, m2: float) -> float:
    return np.sqrt(G * (m1 + m2) / r ** 3)