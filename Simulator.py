import numpy as np
from Propagator import rk4
from Forces import newtonian_gravity
from Config.ConfigClass import SimConfig, BodyConfig

class Simulator:
    def __init__(self, sim_config: SimConfig, state: np.ndarray, mass: np.ndarray, radius: np.ndarray) -> None:
        try:
            self.dt = sim_config.state["dt"]
            self.duration = sim_config.state["duration"]
            self.state = state
            self.radius = radius
            self.mass = mass
            self.num_bodies = np.size(state, axis = 0)
            self.history = np.zeros((self.num_steps + 1, self.num_bodies, 2, 3))
        except KeyError as k:
            print("Initialization of Simulator class went wrong", k)

    @property
    def num_steps(self) -> int:
        return int(self.duration / self.dt)

    def simulate(self) -> None:
        v = self.state[:, 1]
        r = self.state[:, 0]
        self.history[0] = self.state
        for step in range(self.num_steps):
            r_step, v_step = rk4(r, v, self.mass, newtonian_gravity, self.dt)
            self.state[:, 1] = v_step
            self.state[:, 0] = r_step
            self.history[step + 1] = self.state