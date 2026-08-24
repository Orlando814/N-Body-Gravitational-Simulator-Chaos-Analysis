import numpy as np
from lagrange_points import NBodyGravitationalPotentialCalcs
import matplotlib.pyplot as plt

class RestrictedThreeBodyPlot:
    def __init__(self, m1 = 5.9722e24 / 5, m2 = 5.9722e24, g = 6.674 * 10 ** -11,
                 dist_between_bodies = 10):
        self.mass = np.array([m1, m2])
        self.g = g
        self.dist_between_bodies = dist_between_bodies
        self.body_pos = np.array([np.array([m2 / (m1 + m2) * dist_between_bodies, 0.0, 0.0]),
                                  np.array([-m1 / (m1 + m2) * dist_between_bodies, 0.0, 0.0])])

    def run(self, xy_values = 15, point_num = 1000):
        x = np.linspace(-xy_values, xy_values, point_num)
        y = np.linspace(-xy_values, xy_values, point_num)
        x_mesh, y_mesh = np.meshgrid(x, y)
        points = np.array([x_mesh.ravel(), y_mesh.ravel(), np.zeros_like(x_mesh).ravel()])

        n_body_grav_potentials = NBodyGravitationalPotentialCalcs(self.body_pos, self.mass, self.g)
        z_pot, lagrange_points, lagrange_labels = (
            n_body_grav_potentials.restricted_three_body_prob_plot_vals(points, self.dist_between_bodies))
        z_pot = z_pot.reshape(x_mesh.shape)

        fig = plt.figure(figsize = (14, 6))
        ax_3d = fig.add_subplot(1, 2, 1, projection = "3d")
        ax_con = fig.add_subplot(1, 2, 2)
        surf = ax_3d.plot_surface(x_mesh, y_mesh, z_pot, cmap = "viridis")
        contour_lines = ax_con.contour(x_mesh, y_mesh, z_pot, levels = 30, colors = "white", linewidths = 0.7)
        contour = ax_con.contourf(x_mesh, y_mesh, z_pot, levels = 20, cmap = "viridis")
        lagrange_points_scatter = ax_con.scatter(lagrange_points[:, 0], lagrange_points[:, 1],
                                                 color = "red", s = 100, marker = "*")
        for point in range(np.size(lagrange_points, axis = 0)):
            ax_con.annotate(lagrange_labels[point],
                            (lagrange_points[point][0], lagrange_points[point][1]),
                            xytext = (5, 5),
                            textcoords = "offset points",
                            color = "black",
                            fontsize = 11)
        fig.colorbar(surf, ax = ax_3d, shrink = 0.8)
        fig.colorbar(surf, ax = ax_con, shrink = 0.8)
        plt.tight_layout()
        plt.show()
