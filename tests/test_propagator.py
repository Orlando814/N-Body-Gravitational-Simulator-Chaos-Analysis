import pytest
import numpy as np

@pytest.fixture
def s():
    r = np.array([[0, 0, 0], [500, -600, 20], [60, -725, -250]])
    v = np.array([[-40, 20, 100], [600, 1000, -867], [345, -654, -234]])
    s = np.zeros((3, 2, 3))
    s[:, 0] = r
    s[:, 1] = v
    return s