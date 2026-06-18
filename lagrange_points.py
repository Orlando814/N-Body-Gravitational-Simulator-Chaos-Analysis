import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.optimize import brentq

def two_body_potential_gradient_rot(r: np.ndarray, pos: np.ndarray, mass: np.ndarray,
                                    omega: float, g: float, rot_boolean: bool) -> np.ndarray:
    if rot_boolean:
        centrifugal_x = omega ** 2 * r[0]
        centrifugal_y = omega ** 2 * r[1]
    else:
        centrifugal_x = 0
        centrifugal_y = 0

    coef = g * -1 / 2 * mass

    deriv = coef[:, np.newaxis] * ((r[0] - pos[:, 0, np.newaxis]) ** 2 + (r[1] - pos[:, 1, np.newaxis]) ** 2 + (r[2] - pos[:, 2, np.newaxis]) ** 2) ** (-3 / 2)

    du_dx = np.sum(-deriv * 2 * (r[0] - pos[:, 0, np.newaxis]), axis = 0) - centrifugal_x
    du_dy = np.sum(-deriv * 2 * (r[1] - pos[:, 1, np.newaxis]), axis = 0) - centrifugal_y
    du_dz = np.sum(-deriv * 2 * (r[2] - pos[:, 2, np.newaxis]), axis = 0)
    return np.array([du_dx, du_dy, du_dz])

def two_body_potential_rot(r: np.ndarray, pos: np.ndarray, mass: np.ndarray, omega: float, g: float,
                           rot_boolean: bool) -> np.ndarray:
    if rot_boolean:
        centrifugal = 1 / 2 * omega ** 2 * (r[0] ** 2 + r[1] ** 2)
    else:
        centrifugal = 0

    coef = g * mass

    grav_potential = coef[:, np.newaxis] * ((r[0] - pos[:, 0, np.newaxis]) ** 2 + (r[1] - pos[:, 1, np.newaxis]) ** 2 + (r[2] - pos[:, 2, np.newaxis]) ** 2) ** (-1 / 2)

    return np.sum(-grav_potential, axis = 0) - centrifugal

def angular_velocity(r: float, m1: float, m2: float, g: float) -> float:
    num = g * (m1 + m2)
    denom = r ** 3
    return np.sqrt(num / denom)

# This finds L1-3 numerically using SciPy's brentq. We assume that all three points are on the x axis and y = 0 to avoid
# using fsovle for simplicity. The way we find these points is scanning for a sign flip in the gradient across the
# x-axis and filtering out any invalid sign flips due to passing over the bodies. Also assumes that there will only be
# three lagrange points returned which is why I created a static array that can contain 3 points only.
def find_lagrange_123(gradient_func, b1_pos: np.ndarray, b2_pos: np.ndarray, m1: float, m2: float, omega: float,
                      g: float, dist_bet_bodies: float) -> np.ndarray:
    lagrange_points123 = np.zeros((3, 2))
    lagrange_points123[:, 1] = 0

    # Creates the grid to iterate over the two bodies
    grid = np.arange(-dist_bet_bodies * 10, dist_bet_bodies * 10, dist_bet_bodies / 300)

    mass = np.array([m1, m2])
    pos = np.array([b1_pos, b2_pos])
    mass_ratio = mass / np.sum(mass, axis = 0)

    # Gradient along the x axis
    grad_x = gradient_func(np.array([grid, np.zeros_like(grid), np.zeros_like(grid)]), pos, mass, omega, g, True)[0]

    # Will find indices for lagrange points, find there positions, then create the bracket to solve for 0
    lagrange_points_inx = correct_lagrange_inx(grid, grad_x, pos, mass_ratio, dist_bet_bodies)
    bracket = np.array([grid[lagrange_points_inx], grid[lagrange_points_inx + 1]]).T
    # dummy function to use in the brentq since brentq can only pass a single value. We also return only the x value of
    # the gradient becuase we already know at a lagrange point y, z = 0 so we just have to solve where x = 0
    def f(axis):
        out = gradient_func(np.array([float(axis), 0.0, 0.0]), pos, mass, omega, g, True)
        return float(out[0].item())

    # We're solving for three points so three iterations. The brentq basically takes a function and a left / right bound
    # These bounds need to be of the opposite sign as that means there is a 0 somewhere in between them. Then the brentq
    # will solve for a x value which results in a 0 for the provided function based on the two bounds
    try:
        for point in range(np.size(bracket, axis = 0)):
            lagrange_points123[point][0] = brentq(f, bracket[point][0], bracket[point][1])
    except IndexError:
        print("Incorrect number of points included")
    return lagrange_points123

# L4-5 form equidistance triangle between the two masses present so x-axis point is just the middle of the bodies.
# Pretty cool
def lagrange_45_closed_form(b1_pos: np.ndarray, b2_pos: np.ndarray) -> np.ndarray:
    avg_dist = (b1_pos[0] + b2_pos[0]) / 2
    actual_dist = abs(b1_pos[0] - b2_pos[0])
    l4 = np.array([avg_dist, np.sqrt(3) / 2 * actual_dist])
    l5 = np.array([avg_dist, -np.sqrt(3) / 2 * actual_dist])
    return np.array([l4, l5])

# Will remove incorrect lagrange points when solving for 0 in each axis as the position of a body will be flagged as a
# lagrange point. Main idea of this func is to find brackets for a 0 point by first identifying potential point and
# removing it if it's close to a body
def correct_lagrange_inx(test_pos: np.ndarray, grad_axis: np.ndarray, pos_body: np.ndarray, mass_ratio: np.ndarray, dist_bet_bodies: float) -> np.ndarray:
    # The result will only be negative if there as been a sign flip (- * +)
    sign_flip = grad_axis[1:] * grad_axis[:-1] < 0
    sign_flip_inx = np.where(sign_flip)[0]
    sign_flip_pos = test_pos[sign_flip_inx]

    # check if the distance is to close to the body considering each bodies mass / dist between bodies
    distance_to_body_check = (abs(sign_flip_pos - pos_body[:, 0, np.newaxis])
                              > dist_bet_bodies * mass_ratio[:, np.newaxis] * 0.1)

    # Check if the magnitude of the gradient at these flipping points is unreasonably
    gradient_idx = abs(grad_axis[sign_flip_inx])
    huge_mag_check = gradient_idx < np.median(np.clip(gradient_idx, None, np.percentile(gradient_idx, 95))) * 3

    # Collapses booleans lists for each body through and a and statemnet so all cords for all bodies have to return
    # True. Then it merges this with the other magnitude check.
    overall_check = np.all(distance_to_body_check, axis = 0) & huge_mag_check

    # Filter ou any incorrect points
    return sign_flip_inx[overall_check]

def create_lagrange_labels(lagrange_points: np.ndarray, r1: np.ndarray,  r2: np.ndarray,
                           m1: float, m2: float) -> list[str]:
    labels = []
    if (m1 > m2):
        heavy_body = r1
        light_body = r2
    else:
        heavy_body = r2
        light_body = r1
    for point in lagrange_points:

        # L5 point has negative y value
        if point[1] < 0:
            labels.append("L5")

        # L4 point has positive y value
        elif point[1] > 0:
            labels.append("L4")

        # L1 is inbetween the two bodies
        elif ((heavy_body[0] < point[0]) and (point[0] < light_body[0])) or ((heavy_body[0] > point[0]) and (point[0] > light_body[0])):
            labels.append("L1")

        # L3 is the farthest away from the light body
        elif np.linalg.norm(light_body[:-1] - point) == max(np.linalg.norm(light_body[:-1] - lagrange_points,
                                                            axis = 1)):
            labels.append("L3")

        # L2 is the farthest away from the heavy body
        elif np.linalg.norm(heavy_body[:-1] - point) == max(np.linalg.norm(heavy_body[:-1] - lagrange_points,
                                                                           axis = 1)):
            labels.append("L2")
    return labels

# m1, m2, g = 5.9722e24 / 5, 5.9722e24, 6.674 * 10 ** -11
# dist_between_bodies = 10
# r1 = np.array([-m2 / (m1 + m2) * dist_between_bodies, 0.0, 0.0])
# r2 = np.array([m1 / (m1 + m2) * dist_between_bodies, 0.0, 0.0])
# omega = angular_velocity(dist_between_bodies, m1, m2, g)
# x = np.linspace(-15, 15, 1000)
# y = np.linspace(-15, 15, 1000)
#
# x_mesh, y_mesh = np.meshgrid(x, y)
# points = np.array([x_mesh.ravel(), y_mesh.ravel(), np.zeros_like(x_mesh).ravel()])
# z_pot = two_body_potential_rot(points, np.array([r1, r2]), np.array([m1, m2]), omega, g, True)
# z_pot = z_pot.reshape(x_mesh.shape)
# z_pot = np.clip(z_pot, np.percentile(z_pot, 1), None)
#
# fig = plt.figure(figsize = (14, 6))
# ax_3d = fig.add_subplot(1, 2, 1, projection = "3d")
# ax_con = fig.add_subplot(1, 2, 2)
# surf = ax_3d.plot_surface(x_mesh, y_mesh, z_pot, cmap = "viridis")
# contour_lines = ax_con.contour(x_mesh, y_mesh, z_pot, levels = 30, colors = "white", linewidths = 0.7)
# contour = ax_con.contourf(x_mesh, y_mesh, z_pot, levels = 20, cmap = "viridis")
# lagrange_123 = find_lagrange_123(two_body_potential_gradient_rot, r1, r2, m1, m2, omega, g, dist_between_bodies)
# lagrange_45 = lagrange_45_closed_form(r1, r2)
# lagrange_points = np.concatenate([lagrange_123, lagrange_45], axis = 0)
# lagrange_points_scatter = ax_con.scatter(lagrange_points[:, 0], lagrange_points[:, 1],
#                                          color = "red", s = 100, marker = "*")
# lagrange_labels = create_lagrange_labels(lagrange_points, r1, r2, m1, m2)
# for point in range(np.size(lagrange_points, axis = 0)):
#     ax_con.annotate(lagrange_labels[point],
#                     (lagrange_points[point][0], lagrange_points[point][1]),
#                     xytext = (5, 5),
#                     textcoords = "offset points",
#                     color = "black",
#                     fontsize = 11)
# fig.colorbar(surf, ax = ax_3d, shrink = 0.8)
# fig.colorbar(surf, ax = ax_con, shrink = 0.8)
# plt.tight_layout()
# plt.show()

m1, m2, g, dist_between_bodies = 1, 10, 1, 10
omega = angular_velocity(dist_between_bodies, m1, m2, g)

w = np.linspace(m2 / (m1 + m2) * dist_between_bodies, -m2 / (m1 + m2) * dist_between_bodies, 100)
b = np.linspace(m1 / (m1 + m2) * dist_between_bodies, -m1 / (m1 + m2) * dist_between_bodies, 100)


r1 = np.array([w, np.zeros_like(w), np.zeros_like(w)]).T
r2 = np.array([np.zeros_like(b), np.zeros_like(b), np.zeros_like(b)]).T
r3 = np.array([np.zeros_like(w), w, np.zeros_like(w)]).T
r4 = np.array([np.zeros_like(w), -w, np.zeros_like(w)]).T
r5 = np.array([-w, np.zeros_like(w), np.zeros_like(w)]).T

grid_n = 150
x_length = 11
y_length = 11
x = np.linspace(-x_length, x_length, grid_n)
y = np.linspace(-y_length, y_length, grid_n)
x_mesh, y_mesh = np.meshgrid(x, y, indexing="ij")
points = np.array([x_mesh.ravel(), y_mesh.ravel(), np.zeros_like(x_mesh).ravel()])


app = pg.mkQApp()

win = QtWidgets.QWidget()
layout = QtWidgets.QHBoxLayout(win)

def compute_z(frame: float):
    global norm, z_min, z_max
    frame_new = frame % len(r1)
    z_pot = two_body_potential_rot(points, np.array([
        r1[frame_new],
        r2[frame_new],
        r3[frame_new],
        r4[frame_new],
        r5[frame_new]]),
        np.array([m1, m2, m1, m1, m1]), omega, g, True)
    z_pot = z_pot.reshape(x_mesh.shape)
    z_pot = np.clip(z_pot, np.percentile(z_pot, 1), None)
    z_pot_no_well = np.clip(z_pot, np.percentile(z_pot, 5), None)
    z_min = z_pot_no_well.min()
    z_max = z_pot_no_well.max()
    norm = Normalize(z_min, z_max)
    return z_pot

view = gl.GLViewWidget()
view.setCameraPosition(distance=60, elevation=15, azimuth=-60)
layout.addWidget(view, 1)

plot_2D = pg.PlotWidget()
layout.addWidget(plot_2D, 1)

z0 = compute_z(0)
z_shift = -(z0.max() + z0.min()) / 2
z_min = z0.min()
z_max = z0.max()
norm = Normalize(z_min, z_max)
colors = cm.viridis(norm(z0))
surf = gl.GLSurfacePlotItem(x = x, y = y, z = z0 + z_shift, shader=None, colors = colors,
                            drawEdges = True, edgeColor = (0.4,0.4,0.4,0.3))
surf.setGLOptions('translucent')
view.addItem(surf)

contour = pg.ImageItem()
plot_2D.addItem(contour)
plot_2D.setLabel('left', 'y')
plot_2D.setLabel('bottom', 'x')
contour.setImage(z0, levels = (z_min, z_max), autoLevels = False)
contour.setColorMap(pg.colormap.get("viridis"))
contour.setRect(QtCore.QRectF(-x_length, -y_length, x_length * 2, y_length * 2))



frame = 0
def update():
    global frame, colors, norm
    frame += 1
    z = compute_z(frame)
    colors = cm.viridis(norm(z))
    surf.setData(z = z + z_shift, colors = colors, edgeColor = (0.4,0.4,0.4,0.3))
    contour.setImage(z, levels = (z_min, z_max), autoLevels = False)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(16)

win.resize(1200, 600)
win.show()

pg.exec()