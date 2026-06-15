import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def two_body_potential_gradient_rot(r: np.ndarray, pos: np.ndarray, mass: np.ndarray,
                                    omega: float, g: float) -> np.ndarray:
# def two_body_potential_gradient_rot(r: np.ndarray, b1_pos, b2_pos, m1, m2,
#                                     omega: float, g: float) -> np.ndarray:
    coef = g * -1 / 2 * mass

    deriv = -((r[0] - pos[:, 0]) ** 2 + (r[1] - pos[:, 1]) ** 2 + (r[2] - pos[:, 2]) ** 2) ** (-3 / 2)

    du_dx = -coef * deriv * 2 * (r[0] - pos[:, 0]) - omega ** 2 * r[0]
    du_dy = -coef * deriv * 2 * (r[1] - pos[:, 1]) - omega ** 2 * r[1]
    du_dz = -coef * deriv * 2 * (r[2] - pos[:, 2])

    # body1_deriv = g * -1 / 2 * m1 * ((r[0] - b1_pos[0]) ** 2 + (r[1] - b1_pos[1]) ** 2 + (r[2] - b1_pos[2]) ** 2) ** (
    #         -3 / 2)
    # body2_deriv = g * -1 / 2 * m2 * ((r[0] - b2_pos[0]) ** 2 + (r[1] - b2_pos[1]) ** 2 + (r[2] - b2_pos[2]) ** 2) ** (
    #         -3 / 2)
    # du_dx = -body1_deriv * 2 * (r[0] - b1_pos[0]) - body2_deriv * 2 * (r[0] - b2_pos[0]) - omega ** 2 * r[0]
    # du_dy = -body1_deriv * 2 * (r[1] - b1_pos[1]) - body2_deriv * 2 * (r[1] - b2_pos[1]) - omega ** 2 * r[1]
    # du_dz = -body1_deriv * 2 * (r[2] - b1_pos[2]) - body2_deriv * 2 * (r[2] - b2_pos[2])

    return np.array([np.sum(du_dx, axis = 0), np.sum(du_dy, axis = 0), np.sum(du_dz, axis = 0)])
    # return np.array([du_dx, du_dy, du_dz])

def two_body_potential_rot(r: np.ndarray, pos: np.ndarray, mass: np.ndarray, omega: float, g: float) -> float:
# def two_body_potential_rot(r: np.ndarray, b1_pos, b2_pos, m1, m2,
#                                     omega: float, g: float) -> np.ndarray:

    coef = g * mass

    grav_potential = coef * ((r[0] - pos[:, 0]) ** 2 + (r[1] - pos[:, 1]) ** 2 + (r[2] - pos[:, 2]) ** 2) ** (-1 / 2)

    centrifugal = 1 / 2 * omega ** 2 * (r[0] ** 2 + r[1] ** 2)

    # body1_pot = g * m1 * ((r[0] - b1_pos[0]) ** 2 + (r[1] - b1_pos[1]) ** 2 + (r[2] - b1_pos[2]) ** 2) ** (-1 / 2)
    # body2_pot = g * m2 * ((r[0] - b2_pos[0]) ** 2 + (r[1] - b2_pos[1]) ** 2 + (r[2] - b2_pos[2]) ** 2) ** (-1 / 2)
    # centrifugal = 1 / 2 * omega ** 2 * (r[0] ** 2 + r[1] ** 2)
    # potential = -body1_pot - body2_pot - centrifugal
    # return potential
    return np.sum(-grav_potential, axis = 0) - centrifugal

def angular_velocity(r: float, m1: float, m2: float, g: float) -> float:
    num = g * (m1 + m2)
    denom = r ** 3
    return np.sqrt(num / denom)

def find_lagrange_123(gradient_func, b1_pos: np.ndarray, b2_pos: np.ndarray, m1: float, m2: float, omega: float,
                      g: float, dist_bet_bodies: float) -> np.ndarray:
    lagrange_123 = np.zeros((3, 2))

    # l1, l2, l3 are all on x-axis so y = 0
    lagrange_123[:, 1] = 0

    # Holds the left and right bounds that the brentq will solve 0 for
    point_brackets = np.zeros((3, 2))

    # Creates the grid to iterate over the two bodies
    grid = np.arange(-dist_bet_bodies * 10, dist_bet_bodies * 10, dist_bet_bodies / 300)

    # Couple of vars to store information through the loo
    old_grad = gradient_func(np.array([grid[0], 0, 0]), b1_pos, b2_pos, m1, m2, omega, g)[0]
    old_step = 0
    count = 0
    mass_ratio_b1 = m1 / (m1 + m2)
    mass_ratio_b2 = m2 / (m1 + m2)
    for step in grid:
        new_grad = gradient_func(np.array([step, 0, 0]), b1_pos, b2_pos, m1, m2, omega, g)[0]

        # First check if there has been a sign flip between current grad and previus one. This means that between these
        # two x values there is a potential lagrange point. Also exclude step = 0 as are starting at step = 1
        if ((new_grad > 0 and old_grad < 0) or (new_grad < 0 and old_grad > 0)) and step != grid[0]:

            # When we get close to the bodies there can be a sign flip due to gradient becomeing infty so we filter out
            # any points that are very close to the bodies themselves. The filter comes from mainly the distance between
            # the bodies, but in the case of large bodies 10% bceomes significant so we use the mass ratio to sclae it
            # based on mass as well
            if (abs(step - b1_pos[0]) > dist_bet_bodies * mass_ratio_b1 * 0.1
                    and abs(step - b2_pos[0]) > dist_bet_bodies * mass_ratio_b2 * 0.1):
                point_brackets[count][0] = old_step
                point_brackets[count][1] = step
                count += 1
        old_grad = new_grad
        old_step = step

    # dummy function to use in the brentq since brentq can only pass a single value. We also return only the x value of
    # the gradient becuase we already know at a lagrange point y, z = 0 so we just have to solve where x = 0
    def f(x):
        return gradient_func(np.array([x, 0, 0]), b1_pos, b2_pos, m1, m2, omega, g)[0]

    # We're solving for three points so three iterations. The brentq basically takes a function and a left / right bound
    # These bounds need to be of the opposite sign as that means there is a 0 somewhere in between them. Then the brentq
    # will solve for a x value which results in a 0 for the provided function based on the two bounds
    for point in range(count):
        lagrange_123[point][0] = brentq(f, point_brackets[point][0], point_brackets[point][1])
    return lagrange_123

def find_lagrange_45(dist_bet_bodies: float, m1: float, m2: float) -> np.ndarray:
    l4 = np.array([dist_bet_bodies / 2 * (m1 - m2) / (m1 + m2), np.sqrt(3) / 2 * dist_bet_bodies])
    l5 = np.array([dist_bet_bodies / 2 * (m1 - m2) / (m1 + m2), -np.sqrt(3) / 2 * dist_bet_bodies])
    return np.array([l4, l5])

m1, m2, g = 1, 10, 1
dist_between_bodies = 10
r1 = np.array([m2 / (m1 + m2) * dist_between_bodies, 0.0, 0.0])
r2 = np.array([-m1 / (m1 + m2) * dist_between_bodies, 0.0, 0.0])
omega = angular_velocity(dist_between_bodies, m1, m2, g)
x = np.linspace(-15, 15, 1000)
y = np.linspace(-15, 15, 1000)

# generalized version works for gradient
# o = two_body_potential_gradient_rot(np.array([0, 0, 0]), np.array([r1, r2]), np.array([m1, m2]), omega, g)
# o = two_body_potential_gradient_rot(np.array([0, 0, 0]), r1, r2, m1, m2, omega, g)
# 0,12.08790
# 1,0.00000
# 2,0.00000

o = two_body_potential_rot(np.array([0, 0, 0]), np.array([r1, r2]), np.array([m1, m2]), omega, g)
# o = two_body_potential_rot(np.array([0, 0, 0]), r1, r2, m1, m2, omega, g)
# -11.109999999999998
# -11.109999999999998

x_mesh, y_mesh = np.meshgrid(x, y)
points = np.array([x_mesh, y_mesh, np.zeros_like(x_mesh)])
z_pot = two_body_potential_rot(points, r1, r2, m1, m2, omega, g)
z_pot = np.clip(z_pot, np.percentile(z_pot, 1), None)

fig = plt.figure(figsize = (14, 6))
ax_3d = fig.add_subplot(1, 2, 1, projection = "3d")
ax_con = fig.add_subplot(1, 2, 2)
surf = ax_3d.plot_surface(x_mesh, y_mesh, z_pot, cmap = "viridis")
contour_lines = ax_con.contour(x_mesh, y_mesh, z_pot, levels = 30, colors = "white", linewidths = 0.7)
contour = ax_con.contourf(x_mesh, y_mesh, z_pot, levels = 20, cmap = "viridis")
lagrange_123 = find_lagrange_123(two_body_potential_gradient_rot, r1, r2, m1, m2, omega, g, dist_between_bodies)
lagrange_45 = find_lagrange_45(dist_between_bodies, m2, m1)
lagrange_points = np.concatenate([lagrange_123, lagrange_45], axis = 0)
lagrange_points_scatter = ax_con.scatter(lagrange_points[:, 0], lagrange_points[:, 1],
                                         color = "red", s = 100, marker = "*")
lagrange_labels = ["L3", "L1", "L2", "L4", "L5"]
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
