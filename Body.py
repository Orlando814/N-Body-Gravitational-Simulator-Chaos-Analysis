from Config.ConfigClass import BodyConfig
import numpy as np

class Body:
    def __init__(self, body_config: BodyConfig):
      try:
          self.r = np.array(body_config.state["r"])
          self.v = np.array(body_config.state["v"])
          self.radius = body_config.state["radius"]
          self.mass = body_config.state["mass"]
      except KeyError as k:
          print("Initialization of Body class went wrong", k)

    @property
    def speed(self):
        return np.linalg.norm(self.v)

    @property
    def diameter(self):
        return self.radius * 2



