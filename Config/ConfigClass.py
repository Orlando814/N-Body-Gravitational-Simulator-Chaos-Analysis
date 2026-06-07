from GenericConts import load_yaml
import numpy as np

body_config_path = "three_bodies.yaml"
sim_config_path = "SimConfig.yaml"

class Config:
    def __init__(self):
        self.sim = load_yaml(sim_config_path)
        self.body = load_yaml(body_config_path)

class SimConfig:
    def __init__(self):
        config = Config()
        self.state = {}
        for key, value in config.sim.items():
            self.state[key] = value


class BodyConfig:
    def __init__(self, config: Config):
        config = config
        self.state = {}
        self.num_bodies = 0
        for key, value in config.body.items():
            self.state[key] = value
            self.num_bodies += 1
        self.state_array = self.state_array()
        self.mass = self.mass()
        self.radius = self.radius()

    def state_array(self) -> np.ndarray:
        state_array = np.zeros((self.num_bodies, 2,  3))
        count = 0
        for key, value in self.state.items():
            state_array[count] = np.array([np.array(value["r"]), np.array(value["v"])])
            count += 1
        return state_array

    def mass(self) -> np.ndarray:
        mass = np.array([])
        for key, value in self.state.items():
            mass = np.append(mass, value["mass"])
        return mass

    def radius(self) -> np.ndarray:
        radius = np.array([])
        for key, value in self.state.items():
            radius = np.append(radius, value["radius"])
        return radius
