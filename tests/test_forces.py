import pytest
import numpy as np
from Forces import newtonian_gravity
from GenericConts import G, EARTH_MASS


@pytest.fixture
def constants():
    r = np.array([[0, 0, 0], [10000, -10000, 100000], [-100000, 20000, 56000]], dtype=float)
    mass = np.array([EARTH_MASS, EARTH_MASS * 2, EARTH_MASS * 0.5])
    return r, mass


def test_newtonian_gravity(constants):
    r, mass = constants
    body_one_a = G * mass[1] / (np.linalg.norm(r[0] - r[1])) ** 2 * ((r[1] - r[0]) / np.linalg.norm(r[1] - r[0])) + (
                         G * mass[2] / (np.linalg.norm(r[0] - r[2])) ** 2 * (r[2] - r[0]) / np.linalg.norm(r[2] - r[0]))
    body_two_a = G * mass[0] / (np.linalg.norm(r[1] - r[0])) ** 2 * (r[0] - r[1]) / np.linalg.norm(r[0] - r[1]) + (
            G * mass[2] / (np.linalg.norm(r[1] - r[2])) ** 2 * (r[2] - r[1]) / np.linalg.norm(r[2] - r[1]))
    body_three_a = G * mass[0] / (np.linalg.norm(r[2] - r[0])) ** 2 * (r[0] - r[2]) / np.linalg.norm(r[0] - r[2]) + (
            G * mass[1] / (np.linalg.norm(r[2] - r[1])) ** 2 * (r[1] - r[2]) / np.linalg.norm(r[1] - r[2]))
    accel_result = np.array([body_one_a, body_two_a, body_three_a])
    accel = newtonian_gravity(r, mass)
    assert np.allclose(accel_result, accel, atol=1e-100) is True
    assert np.allclose(accel_result, accel, rtol=1e-100) is True