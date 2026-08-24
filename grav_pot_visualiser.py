from Config.ConfigClass import Config
from Simulator import *
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from GenericConts import load_yaml
from lagrange_points import NBodyGravitationalPotentialCalcs

class NBodyGravitationalPotentialVisualiser:
    def __init__(self, initial_conditions = "six_bodies", point_density = 120, g = 1) -> None:
        self.initial_conditions = initial_conditions
        self.point_density = point_density
        self.g = g
        self.sim_hist, self.mass = self.create_sim_history()
        self.normalize_sim_history()
        self.x_min = float(np.min(self.sim_hist[0, :, 0, 0]))
        self.x_max = float(np.max(self.sim_hist[0, :, 0, 0]))
        self.y_min = float(np.min(self.sim_hist[0, :, 0, 1]))
        self.y_max = float(np.max(self.sim_hist[0, :, 0, 1]))
        self.z_min = float(np.min(self.sim_hist[0, :, 0, 2]))
        self.z_max = float(np.max(self.sim_hist[0, :, 0, 2]))
        self.norm = Normalize(self.z_min, self.z_max)
        self.frame = 0
        self.surf = None
        self.contour = None
        self.z_shift = -1

    def create_sim_history(self):
        config = Config()
        config.body = load_yaml(self.initial_conditions + ".yaml")
        body_config = BodyConfig(config)
        state = body_config.state_array
        mass = body_config.mass
        radius = body_config.radius
        sim = SimConfig()
        simulator = Simulator(sim, state, mass, radius)
        simulator.simulate()
        return simulator.history, mass

    # Finding maximum distance between starting bodies and then normalizing their distance in the history to make things
    # easier
    def normalize_sim_history(self):
        initial_body_pos = self.sim_hist[0, :, 0]
        dist_bet_bodies_vec = initial_body_pos - initial_body_pos[:, np.newaxis]
        dist_bet_bodies_scal = np.linalg.norm(dist_bet_bodies_vec, axis = 2)
        max_dist_bet_bodies = np.max(dist_bet_bodies_scal)

        # Normalizing body positions / mass through all timesteps
        self.sim_hist[:, :, 0] = self.sim_hist[:, :, 0] / max_dist_bet_bodies
        self.mass = self.mass / np.sum(self.mass)

    def update_xy_min_max(self, frame_new):
        self.x_min = min(self.x_min, float(np.min(self.sim_hist[frame_new, :, 0, 0])))
        self.x_max = max(self.x_max, float(np.max(self.sim_hist[frame_new, :, 0, 0])))
        self.y_min = min(self.y_min, float(np.min(self.sim_hist[frame_new, :, 0, 1])))
        self.y_max = max(self.y_max, float(np.max(self.sim_hist[frame_new, :, 0, 1])))
        self.x_min = min(self.x_min, self.y_min)
        self.y_min = self.x_min
        self.x_max = max(self.x_max, self.y_max)
        self.y_max = self.x_max

    def compute_z(self):
        frame_new = self.frame % np.size(self.sim_hist, axis = 0)
        self.update_xy_min_max(frame_new)
        grid_n = self.point_density
        x_pad = 0.3 * (self.x_max - self.x_min)
        y_pad = 0.3 * (self.y_max - self.y_min)
        x = np.linspace(self.x_min - x_pad, self.x_max + x_pad, grid_n)
        y = np.linspace(self.y_min - y_pad, self.y_max + y_pad, grid_n)
        x_mesh, y_mesh = np.meshgrid(x, y, indexing = "ij")
        points = np.array([x_mesh.ravel(), y_mesh.ravel(), np.zeros_like(x_mesh).ravel()])
        n_body_grav_pot = NBodyGravitationalPotentialCalcs(self.sim_hist[frame_new, :, 0, :], self.mass, self.g)
        z_pot = n_body_grav_pot.n_body_potential_rot(points, 0, False)
        z_pot = z_pot.reshape(x_mesh.shape)
        z_pot = np.clip(z_pot, np.percentile(z_pot, 1), None)
        self.z_min = z_pot.min()
        self.z_max = z_pot.max()
        self.norm = Normalize(self.z_min, self.z_max)
        return x, y, z_pot

    def run(self):
        app = pg.mkQApp()
        win = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(win)

        x, y, z0 = self.compute_z()
        self.z_shift = -(z0.max() + z0.min()) / 2
        self.z_min = z0.min()
        self.z_max = z0.max()
        self.norm = Normalize(self.z_min, self.z_max)
        colors = cm.viridis(self.norm(z0))

        view = gl.GLViewWidget()
        view.setCameraPosition(distance = 20, elevation = 30, azimuth = -60)
        layout.addWidget(view, 1)

        self.surf = gl.GLSurfacePlotItem(x = x, y = y, z = z0 + self.z_shift, shader = None, colors = colors,
                                    drawEdges = True, edgeColor = (0.4, 0.4, 0.4, 0.4))
        self.surf.setGLOptions('translucent')

        view.addItem(self.surf)

        plot_2D = pg.PlotWidget()
        layout.addWidget(plot_2D, 1)

        self.contour = pg.ImageItem()
        plot_2D.addItem(self.contour)
        plot_2D.setLabel('left', 'y')
        plot_2D.setLabel('bottom', 'x')
        self.contour.setImage(z0, levels = (self.z_min, self.z_max), autoLevels = False)
        self.contour.setColorMap(pg.colormap.get("viridis"))
        self.contour.setRect(QtCore.QRectF(np.min(x), np.min(y), np.max(x) - np.min(x), np.max(y) - np.min(y)))

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(32)
        win.resize(1200, 600)
        win.show()
        pg.exec()

    def update(self):
        if self.z_shift == -1 or self.surf is None or self.contour is None:
            raise ValueError("Please use the run function first")
        self.frame += 1
        x, y, z = self.compute_z()
        colors = cm.viridis(self.norm(z))
        self.surf.setData(x = x, y = y, z = z + self.z_shift, colors = colors, edgeColor = (0.4, 0.4, 0.4, 0.4))
        self.contour.setImage(z, levels = (self.z_min, self.z_max), autoLevels = False)
        self.contour.setRect(QtCore.QRectF(np.min(x), np.min(y), np.max(x) - np.min(x), np.max(y) - np.min(y)))