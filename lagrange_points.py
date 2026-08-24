from traitlets.config import ArgumentError
import numpy as np
from scipy.optimize import brentq

class NBodyGravitationalPotentialCalcs:
    def __init__(self, body_pos: np.ndarray, mass: np.ndarray, g: float) -> None:
        self.body_pos = body_pos
        self.mass = mass
        self.g = g

    # Remember that z_pot needs to be .reshape(x_mesh.shape) before using it to plot correctly
    def restricted_three_body_prob_plot_vals(self, points: np.ndarray, dist_between_bodies: float):
        if np.size(self.body_pos, axis = 0) != 2:
            raise ArgumentError("The number of bodies provided is not 2, but " + str(np.size(self.body_pos, axis = 0)))
        omega = self.two_body_ang_vel(dist_between_bodies)
        z_pot = self.n_body_potential_rot(points, omega,True)
        z_pot = np.clip(z_pot, np.percentile(z_pot, 1), None)
        lagrange_123 = self.find_lagrange_123(self.n_body_potential_gradient_rot, omega, dist_between_bodies)
        lagrange_45 = self.lagrange_45_closed_form()
        lagrange_points = np.concatenate([lagrange_123, lagrange_45], axis = 0)
        lagrange_labels = self.create_lagrange_labels(lagrange_points)
        return z_pot, lagrange_points, lagrange_labels

    def n_body_potential_gradient_rot(self, points: np.ndarray, omega: float, rot_coord_bool: bool) -> np.ndarray:
        if rot_coord_bool:
            centrifugal_x = omega ** 2 * points[0]
            centrifugal_y = omega ** 2 * points[1]
        else:
            centrifugal_x = 0
            centrifugal_y = 0

        coef = self.g * -1 / 2 * self.mass

        deriv = coef[:, np.newaxis] * ((points[0] - self.body_pos[:, 0, np.newaxis]) ** 2 +
                                       (points[1] - self.body_pos[:, 1, np.newaxis]) ** 2 +
                                       (points[2] - self.body_pos[:, 2, np.newaxis]) ** 2) ** (-3 / 2)

        du_dx = np.sum(-deriv * 2 * (points[0] - self.body_pos[:, 0, np.newaxis]), axis = 0) - centrifugal_x
        du_dy = np.sum(-deriv * 2 * (points[1] - self.body_pos[:, 1, np.newaxis]), axis = 0) - centrifugal_y
        du_dz = np.sum(-deriv * 2 * (points[2] - self.body_pos[:, 2, np.newaxis]), axis = 0)
        return np.array([du_dx, du_dy, du_dz])

    def n_body_potential_rot(self, points: np.ndarray, omega: float, rot_coord_bool: bool) -> np.ndarray:
        if rot_coord_bool:
            centrifugal = 1 / 2 * omega ** 2 * (points[0] ** 2 + points[1] ** 2)
        else:
            centrifugal = 0

        coef = self.g * self.mass

        grav_potential = coef[:, np.newaxis] * (
                (points[0] - self.body_pos[:, 0, np.newaxis]) ** 2 +
                (points[1] - self.body_pos[:, 1, np.newaxis]) ** 2 + (
                points[2] - self.body_pos[:, 2, np.newaxis]) ** 2) ** (-1 / 2)

        return np.sum(-grav_potential, axis = 0) - centrifugal

    def two_body_ang_vel(self, dist_bet_bodies: float) -> float:
        num = self.g * (self.mass[0] + self.mass[1])
        denom = dist_bet_bodies ** 3
        return np.sqrt(num / denom)

    # This finds L1-3 numerically using SciPy's brentq. We assume that all three points are on the x axis and y = 0 to avoid
    # using fsovle for simplicity. The way we find these points is scanning for a sign flip in the gradient across the
    # x-axis and filtering out any invalid sign flips due to passing over the bodies. Also assumes that there will only be
    # three lagrange points returned which is why I created a static array that can contain 3 points only.
    def find_lagrange_123(self, gradient_func, omega: float, dist_bet_bodies: float) -> np.ndarray:

        lagrange_points123 = np.zeros((3, 2))
        lagrange_points123[:, 1] = 0

        # Creates the grid to iterate over the two bodies
        grid = np.arange(-dist_bet_bodies * 10, dist_bet_bodies * 10, dist_bet_bodies / 300)

        mass_ratio = self.mass / np.sum(self.mass, axis = 0)

        # Gradient along the x-axis
        grad_x = gradient_func(np.array([grid, np.zeros_like(grid), np.zeros_like(grid)]), omega, True)[0]

        # Will find indices for lagrange points, find there positions, then create the bracket to solve for 0
        lagrange_points_inx = self.correct_lagrange_inx(grid, grad_x, self.body_pos, mass_ratio, dist_bet_bodies)
        bracket = np.array([grid[lagrange_points_inx], grid[lagrange_points_inx + 1]]).T

        # dummy function to use in the brentq since brentq can only pass a single value. We also return only the x value of
        # the gradient becuase we already know at a lagrange point y, z = 0 so we just have to solve where x = 0
        def f(axis):
            out = gradient_func(np.array([float(axis), 0.0, 0.0]), omega, True)
            return float(out[0].item())

        # We're solving for three points so three iterations. The brentq basically takes a function and a left / right bound
        # These bounds need to be of the opposite sign as that means there is a 0 somewhere in between them. Then the brentq
        # will solve for an x value which results in a 0 for the provided function based on the two bounds
        try:
            for point in range(np.size(bracket, axis = 0)):
                lagrange_points123[point][0] = brentq(f, bracket[point][0], bracket[point][1])
        except IndexError:
            print("Incorrect number of points included: " + str(np.size(bracket, axis = 0)))
        return lagrange_points123

    # L4-5 form equidistance triangle between the two masses present so x-axis point is just the middle of the bodies.
    # Pretty cool
    def lagrange_45_closed_form(self) -> np.ndarray:
        avg_dist = (self.body_pos[0][0] + self.body_pos[1][0]) / 2
        actual_dist = abs(self.body_pos[0][0] - self.body_pos[1][0])
        l4 = np.array([avg_dist, np.sqrt(3) / 2 * actual_dist])
        l5 = np.array([avg_dist, -np.sqrt(3) / 2 * actual_dist])
        return np.array([l4, l5])

    # Will remove incorrect lagrange points when solving for 0 in each axis as the position of a body will be flagged as a
    # lagrange point. Main idea of this func is to find brackets for a 0 point by first identifying potential point and
    # removing it if it's close to a body
    @staticmethod
    def correct_lagrange_inx(test_pos: np.ndarray, grad_axis: np.ndarray, pos_body: np.ndarray, mass_ratio: np.ndarray,
                             dist_bet_bodies: float) -> np.ndarray:
        # The result will only be negative if there has been a sign flip (- * +)
        sign_flip = grad_axis[1:] * grad_axis[:-1] < 0
        sign_flip_inx = np.where(sign_flip)[0]
        sign_flip_pos = test_pos[sign_flip_inx]

        # check if the distance is to close to the body considering each bodies mass / dist between bodies
        distance_to_body_check = (abs(sign_flip_pos - pos_body[:, 0, np.newaxis])
                                  > dist_bet_bodies * mass_ratio[:, np.newaxis] * 0.1)

        # Check if the magnitude of the gradient at these flipping points is unreasonably
        gradient_idx = abs(grad_axis[sign_flip_inx])
        huge_mag_check = gradient_idx < np.median(np.clip(gradient_idx, None, np.percentile(gradient_idx, 95))) * 3

        # Collapses booleans lists for each body through and a and statement so all cords for all bodies have to return
        # True. Then it merges this with the other magnitude check.
        overall_check = np.all(distance_to_body_check, axis = 0) & huge_mag_check

        # Filter out any incorrect points
        return sign_flip_inx[overall_check]

    # Goes through each Lagrange point and based on the unique conditions of each point, it appends that point's
    # Lagrange number in a list whose label indices match the point indices
    def create_lagrange_labels(self, lagrange_points: np.ndarray) -> list[str]:
        labels = []
        if self.mass[0] > self.mass[1]:
            heavy_body = self.body_pos[0]
            light_body = self.body_pos[1]
        else:
            heavy_body = self.body_pos[1]
            light_body = self.body_pos[0]
        for point in lagrange_points:

            # L5 point has negative y value
            if point[1] < 0:
                labels.append("L5")

            # L4 point has positive y value
            elif point[1] > 0:
                labels.append("L4")

            # L1 is in between the two bodies
            elif ((heavy_body[0] < point[0]) and (point[0] < light_body[0])) or (
                    (heavy_body[0] > point[0]) and (point[0] > light_body[0])):
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