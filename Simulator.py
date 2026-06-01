from Config.ConfigClass import SimConfig
from Body import Body
from typing import Callable
import numpy as np

class Simulator:
    def __init__(self, sim_config: SimConfig, r: np.ndarray, v: np.ndarray, radius: np.ndarray, mass: np.ndarray) -> None:
        try:
            self.dt = sim_config.state["dt"]
            self.num_steps = sim_config.state["num_steps"]
            self.r = r
            self.v = v
            self.radius = radius
            self.mass = mass
            self.num_bodies = r.size
        except KeyError as k:
            print("Initialization of Simulator class went wrong", k)

    @property
    def history(self) -> dict:
        history = {}
        for index in range(self.num_bodies):
            history["body" + str(index + 1)] = {"r": np.zeros(self.num_steps), "v": np.zeros(self.num_steps)}
        return history

    @property
    def duration(self) -> float:
        return self.dt * self.num_steps

    def simulate(self, step_func: Callable[[Body, Body, Callable[[np.ndarray, np.ndarray], np.ndarray]], None]) -> None: # probably needs a state as well
        current_time = 0
        for step in range(self.num_steps):



            current_time += self.dt