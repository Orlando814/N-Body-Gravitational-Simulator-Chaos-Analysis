import yaml

def load_yaml(name: str) -> dict:
    with open(name, 'r') as f:
        return yaml.safe_load(f)

body_config_path = "FigureEight.yaml"
sim_config_path = "SimConfig.yaml"

class Config:
    def __init__(self):
        self.sim = SimConfig(load_yaml(body_config_path))
        self.body = BodyConfig(load_yaml(sim_config_path))

class SimConfig:
    def __init__(self, sim_config: dict):
        self.state = {}
        for key, value in sim_config.items():
            self.state[key] = value

        # try:
        #     self.dt = simConfig["dt"]
        #     self.numSteps = simConfig["numSteps"]
        #     self.
        # except KeyError as k:
        #     print("SimConfig Class couldn't initialize due to incorrect / lack of key:", k)

class BodyConfig:
    def __init__(self, body_config: dict):
        self.state = {}
        for key, value in body_config.items():
            self.state[key] = value
        # try:
        #     self.x = bodyConfig["initial_position"]["x"]
        #     self.y = bodyConfig["initial_position"]["y"]
        #     self.z = bodyConfig["initial_position"]["z"]
        #     self.vx = bodyConfig["initial_velocity"]["vx"]
        #     self.vy = bodyConfig["initial_velocity"]["vy"]
        #     self.vz = bodyConfig["initial_velocity"]["vz"]
        # except KeyError as k:
        #     print("BodyConfig Class couldn't initialize due to incorrect / lack of key:", k)