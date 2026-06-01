import numpy as np
import pytest
from GenericConts import dist_of_all_bodies, direc_of_body_to_other_body


@pytest.fixture
def r() -> np.ndarray:
    return np.array([[3, 4, 0], [1, 2, 2], [2, 3, 6]])  # magnitudes: 5, 3, 7


def test_dist_of_all_bodies(r):
    dist = dist_of_all_bodies(r)
    print(dist)
    result = np.array([[np.linalg.norm(r[0] - r[0]), np.linalg.norm(r[0] - r[1]), np.linalg.norm(r[0] - r[2])],
                       [np.linalg.norm(r[1] - r[0]), np.linalg.norm(r[1] - r[1]), np.linalg.norm(r[1] - r[2])],
                       [np.linalg.norm(r[2] - r[0]), np.linalg.norm(r[2] - r[1]), np.linalg.norm(r[2] - r[2])]
                       ])
    assert np.array_equal(dist, result) is True


def test_direc_of_body_to_other_body(r):
    direc = direc_of_body_to_other_body(r)
    result = np.array([
        [[0.0, 0.0, 0.0], [-0.57735027, -0.57735027, 0.57735027], [-0.16222142, -0.16222142, 0.97332853]],
        [[0.57735027, 0.57735027, -0.57735027], [0.0, 0.0, 0.0], [0.23570226, 0.23570226, 0.94280904]],
        [[0.16222142, 0.16222142, -0.97332853], [-0.23570226, -0.23570226, -0.94280904], [0.0, 0.0, 0.0]]
    ])
    assert np.allclose(direc, result, atol=1e-100) is True
    assert np.allclose(direc, result, rtol=1e-100) is True
