import numpy as np
import pytest
from matplotlib import pyplot as plt
from lagrange_points import *
from scipy.optimize import fsolve

@pytest.fixture
def setup():
    m1, m2 = 1, 10
    dist_between_bodies = 10
    r1 = np.array([-m2 / (m1 + m2) * dist_between_bodies, 0.0, 0.0])
    r2 = np.array([m1 / (m1 + m2) * dist_between_bodies, 0.0, 0.0])
    omega = angular_velocity(dist_between_bodies, m1, m2)
    x = np.linspace(-20, 20, 1000)
    y = np.linspace(-20, 20, 1000)
    x_mesh, y_mesh = np.meshgrid(x, y)
    points = np.array([x_mesh, y_mesh, np.zeros_like(x_mesh)])
    gradient = two_body_potential_gradient_rot(points, r1, r2, m1, m2, omega)
    return r1, r2, m1, m2, omega, gradient, x_mesh, y_mesh

def test_potential_gradient(setup):
    r1, r2, m1, m2, omega, gradient, x_mesh, y_mesh = setup
    gradient_mag = np.linalg.norm(gradient, axis = 0)
    gradient_mag_clip = np.clip(gradient_mag, 0, np.percentile(gradient_mag, 95))
    plt.contourf(x_mesh, y_mesh, gradient_mag_clip, levels = 20)

    def grad_xy(p):  # wrap your gradient for a single (x,y) point, z=0
        g = two_body_potential_gradient_rot(np.array([p[0], p[1], 0.0]), r1, r2, m1, m2, omega)
        return [g[0], g[1]]  # want both components = 0

    guesses = [(-5, 0), (5, 0), (-15, 0), (-5, 8.66), (-5, -8.66)]  # eyeball from your contour
    points = [fsolve(grad_xy, g) for g in guesses]
    for p in points:
        plt.scatter(p[0], p[1], color="red", s=40, marker="*")
    # mask = gradient_mag < np.percentile(gradient_mag, 0.1)
    # xlp = x_mesh[mask]
    # ylp = y_mesh[mask]
    # plt.scatter(xlp, ylp, color="red", s=10)
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.show()

