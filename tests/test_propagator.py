import pytest
import numpy as np
from Propagator import rk4
from GenericConts import EARTH_ACCEL
from Forces import newtonian_gravity

# After solving the diff eq for const acceleration (dv/dt = -9.81), these are the solutions for position and velocity:
# Analytical v = vo + a*t / Analytical r = ro + vo*t + (1/2)*a*t^2
# t = dt
def v_analytical(v0, a, dt):
    return v0 + a * dt

def r_analytical(r0, v0, a, dt):
    return r0 + v0 * dt + 0.5 * a * dt ** 2

@pytest.fixture
def consts():
    r = np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1]])
    v = np.array([[1, 1, 1], [-1, -1, -1], [0, 0, 0]])
    dt = 0.1
    return r, v, dt

# We're going to use a dummy function that just returns the acceleration becuase we don't actually care what the func
# does as long as it's a differential eq.
def dummy_func(x, y):
    return EARTH_ACCEL

# We will compare the propogator's results to the analytical one to catch if it's calculating things properly
def test_propagator_results(consts):
    r, v, dt = consts
    a = EARTH_ACCEL
    r_ana = r_analytical(r, v, a, dt)
    v_ana = v_analytical(v, a, dt)
    r_new, v_new = rk4(r, v, np.array([]), dummy_func, dt)
    assert np.allclose(r_ana, r_new)
    assert np.allclose(v_ana, v_new)

# Checks to make sure that propagator returns initial state at time 0
def test_propagator_initialization(consts):
    r, v, dt = consts
    r_new, v_new = rk4(r, v, np.array([]), dummy_func, 0)
    assert np.array_equal(r, r_new)
    assert np.array_equal(v, v_new)


# To test the truncation error for the propagator we're using an undampened harmonic oscillator with initial conditions:
# x(0) = 1, x'(0) = -2 * 3^(1/2)
# func: x'' = -k * x
def harmon_osc_accel(x: float, k: float):
    return -k * x

# Analytical solution for position of the above function
def harmon_osc_pos(omega: float, dt: float):
    return 1 * np.cos(omega * dt) - (2 * np.sqrt(3) / omega) * np.sin(omega * dt)

# Analytical solution for velocity of the above function
def harmon_osc_vel(omega: float, dt: float):
    return -1 * omega * np.sin(omega * dt) - 2 * np.sqrt(3) * np.cos(omega * dt)

# We are going to test if the local truncation error fo the propagator is O(h^5) by taking the ratio of
# h / w and h / (w * 2) which basically means that whever we half the dt the raito of the previous results (dt) and the
# new results (dt / 2) should be 32 since 2^5 = 32 and h / (h / 2^5) = 32
def test_propagator_errors():
    r, v, k, m, dt  = 1, -2 * np.sqrt(3), 4, 1, 0.1 # m, m/s, N/m, kg, s
    omega = np.sqrt(k / m) # sqrt(N) / m

    # First approx
    r_new1, v_new1 = rk4(r, v, k, harmon_osc_accel, dt)
    r_ana1 = harmon_osc_pos(omega, dt)
    v_ana1 = harmon_osc_vel(omega, dt)

    # Second approx
    r_new2, v_new2 = rk4(r, v, k, harmon_osc_accel, dt / 2)
    r_ana2 = harmon_osc_pos(omega, dt / 2)
    v_ana2 = harmon_osc_vel(omega, dt / 2)

    # Third approx
    r_new3, v_new3 = rk4(r, v, k, harmon_osc_accel, dt / 4)
    r_ana3 = harmon_osc_pos(omega, dt / 4)
    v_ana3 = harmon_osc_vel(omega, dt / 4)

    # Errors
    r1_error = abs(r_new1 - r_ana1)
    v1_error = abs(v_new1 - v_ana1)
    r2_error = abs(r_new2 - r_ana2)
    v2_error = abs(v_new2 - v_ana2)
    r3_error = abs(r_new3 - r_ana3)
    v3_error = abs(v_new3 - v_ana3)

    # local truncation errors the lower the dt the more it should approach 32
    local_trunc_error_r1 = r1_error / r2_error
    local_trunc_error_r2 = r2_error / r3_error
    local_trunc_error_v1 = v1_error / v2_error
    local_trunc_error_v2 = v2_error / v3_error
    assert 0 == 1

