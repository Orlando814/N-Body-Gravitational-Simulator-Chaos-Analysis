from vpython import vec, color, sphere, rate, canvas, label, button, slider, menu, arrow, graph, gcurve, wtext
from Simulator import *
from Config.ConfigClass import Config
import numpy as np
from GenericConts import config_names, load_yaml

class NBodyVisualiser:
    def __init__(self):
        self.running = True
        self.reverse = False
        self.step = 0
        self.bodies = []
        self.labels = []
        self.arrows = []
        self.rate_value = 150
        self.body_num = 0
        self.scene = canvas(width = 1200, height = 800, title = "N Body Problem")
        self.body_dist_1 = -1
        self.body_dist_2 = -1
        self.gc = gcurve(color = color.blue)
        self.sim_history = None
        self.body_names = []
        self.config_changed = False
        self.current_config = -1

    def run(self) -> None:
        config = Config()
        state, mass, radius = self.setup_body_config(config)
        simulator = self.setup_sim_hist(state, mass, radius)
        self.sim_history = simulator.history
        self.body_num = np.size(self.sim_history[0, :, 0], axis = 0)
        self.scene.background = color.black
        self.create_bodies_labels_arrows(self.sim_history, radius)
        self.body_names = self.body_name_list()
        self.visual_loop(simulator.dt)

    @staticmethod
    def setup_sim_hist(state, mass, radius):
        simulator = Simulator(SimConfig(), state, mass, radius)
        simulator.simulate()
        return simulator

    @staticmethod
    def setup_body_config(config):
        new_body = BodyConfig(config)
        state = new_body.state_array
        mass = new_body.mass
        radius = new_body.radius
        return state, mass, radius

    def reset_sim(self):
        config = Config()
        config.body = load_yaml(config_names()[self.current_config] + ".yaml")
        state, mass, radius = self.setup_body_config(config)
        self.sim_history = self.setup_sim_hist(state, mass, radius).history
        self.body_num = np.size(self.sim_history, axis = 1)
        for body in self.bodies:
            body.clear_trail()
            body.visible = False
        for arrow in self.arrows:
            arrow.visible = False
        for label in self.labels:
            label.visible = False
        self.body_names = self.body_name_list()
        self.create_bodies_labels_arrows(self.sim_history, radius)
        self.reverse = False
        self.rate_value = 150
        self.body_dist_1 = -1
        self.body_dist_2 = -1
        self.gc.delete()
        self.running = True
        self.step = 0
        self.scene.center = vec(0, 0, 0)
        self.scene.range = 1500000000
        self.scene.autoscale = True
        self.config_changed = False

    def update_position(self, pos_type: str) -> None:
        count = 0
        if pos_type == "body":
            for body in self.bodies:
                body.pos = vec(*self.sim_history[self.step, count, 0, :])
                count += 1
        elif pos_type == "label":
            for body_label in self.labels:
                body_label.pos = vec(*self.sim_history[self.step, count, 0, :])
                count += 1
        elif pos_type == "arrow":
            for body_arrow in self.arrows:
                if self.reverse:
                    body_arrow.axis = -vec(*self.sim_history[self.step, count, 1, :] * 350)
                else:
                    body_arrow.axis = vec(*self.sim_history[self.step, count, 1, :] * 350)
                body_arrow.pos = vec(*self.sim_history[self.step, count, 0, :])
                count += 1

    def visual_loop(self, dt: float) -> None:
        pause_b = button(text = "Pause", pos = self.scene.title_anchor, bind = self.button_event, id = "Pause")
        reverse_b = button(text = "Reverse", pos = self.scene.title_anchor, bind = self.button_event, id = "Reverse")
        button(text = "Reset", pos = self.scene.title_anchor, bind = self.button_event, id = "Reset")
        drag_s = slider(bind = self.slider_event, pos = self.scene.title_anchor, min = 0,
                        max = np.size(self.sim_history, axis = 0), step = 1, id = "Drag")
        rate_s = slider(bind = self.slider_event, pos = self.scene.title_anchor, min = 1, max = 1500, step = 1,
                        id = "Rate")
        follow_m = menu(bind = self.menu_event, pos = self.scene.title_anchor, choices = self.body_names, id = "Follow")
        dist_graph_m1 = menu(bind = self.menu_event, pos = self.scene.title_anchor, choices = self.body_names,
                             id = "Body 1")
        dist_graph_m2 = menu(bind = self.menu_event, pos = self.scene.title_anchor, choices = self.body_names,
                             id = "Body 2")
        menu(bind = self.menu_event, pos = self.scene.title_anchor, choices = config_names(), id = "Current Config")
        graph(title = 'Distance Between Two Bodies', xtitle = 'Time (s)', ytitle = 'Distance (m)', xmin = 0)
        frame_count = wtext(pos = self.scene.title_anchor)
        while True:
            if self.config_changed:
                self.reset_sim()
                pause_b.text = "Pause"
                reverse_b.text = "Reverse"
                drag_s.value = 0
                rate_s.value = self.rate_value
                follow_m.choices = self.body_names
                dist_graph_m1.choices = self.body_names
                dist_graph_m2.choices = self.body_names
            if self.running:
                rate(self.rate_value)
                frame_count.text = "Frame: " + str(self.step) + "/" + str(np.size(self.sim_history, axis = 0))
                if self.step == np.size(self.sim_history, axis = 0):
                    for body in self.bodies:
                        body.clear_trail()
                    self.step = 0
                    self.gc.delete()
                if self.step < 0 and self.reverse:
                    self.step = np.size(self.sim_history, axis = 0) - 1
                    for body in self.bodies:
                        body.clear_trail()
                    self.gc.delete()
                if (self.body_dist_1 >= 0) and (self.body_dist_2 >= 0):
                    self.gc.plot(dt * self.step,
                                 np.linalg.norm(self.sim_history[self.step, self.body_dist_1, 0] -
                                                self.sim_history[self.step, self.body_dist_2, 0]))
                self.update_position("body")
                self.update_position("label")
                self.update_position("arrow")
                if self.reverse:
                    self.step -= 1
                else:
                    self.step += 1

    def create_bodies_labels_arrows(self, sim_hist: np.ndarray, radius: np.ndarray) -> None:
        self.bodies = []
        self.labels = []
        self.arrows = []
        for body in range(self.body_num):
            hue = body / self.body_num
            new_sphere = sphere(pos = vec(*sim_hist[0, body, 0, :]), radius = radius[body],
                                color = color.hsv_to_rgb(vec(hue, 1, 1)),
                                make_trail = True, retain = np.size(sim_hist, axis = 0) * 0.05,
                                trail_radius = radius[body] * 0.15, emissive = True)
            body_label = label(pos = vec(*sim_hist[0, body, 0, :]), xoffset = radius[body] * 0.0000005,
                               yoffset = radius[body] * 0.0000005, text = "Body " + str(body + 1), box = False,
                               line = True,
                               linecolor = color.white, opacity = 0.1)
            body_arrow = arrow(pos = vec(*sim_hist[0, body, 0, :]), axis = vec(*sim_hist[0, body, 1, :] * 250),
                               color = color.hsv_to_rgb(vec(hue, 1, 1)))
            self.bodies.append(new_sphere)
            self.labels.append(body_label)
            self.arrows.append(body_arrow)

    def button_event(self, evt) -> None:
        if evt.id == "Reverse":
            self.reverse = not self.reverse
            if self.reverse:
                evt.text = "Forward"
            else:
                evt.text = "Reverse"
            for body in self.bodies:
                body.clear_trail()
            self.gc.delete()
        elif evt.id == "Pause":
            self.running = not self.running
            if self.running:
                evt.text = "Pause"
            else:
                evt.text = "Run"
        elif evt.id == "Reset":
            self.running = True
            self.reverse = False
            self.step = 0
            self.rate_value = 150
            self.body_dist_1 = -1
            self.body_dist_2 = -1
            self.gc.delete()
            for body in self.bodies:
                body.clear_trail()

    def slider_event(self, evt) -> None:
        if evt.id == "Drag":
            for body in self.bodies:
                body.clear_trail()
            self.gc.delete()
            self.step = evt.value
        elif evt.id == "Rate":
            self.rate_value = evt.value

    def menu_event(self, evt) -> None:
        if evt.id == "Follow":
            if evt.index is None:
                self.scene.camera.follow(None)
            else:
                for body in range(self.body_num):
                    if evt.index - 1 == body:
                        self.scene.camera.follow(self.bodies[body])
        elif evt.id == "Body 1":
            if evt.index is None:
                self.body_dist_1 = -1
            else:
                self.body_dist_1 = evt.index - 1
        elif evt.id == "Body 2":
            if evt.index is None:
                self.body_dist_2 = -1
            else:
                self.body_dist_2 = evt.index - 1
        elif evt.id == "Current Config":
            self.current_config = evt.index
            self.config_changed = True

    def body_name_list(self) -> list:
        names = ["None"]
        for body in range(self.body_num):
            names.append("Body " + str(body + 1))
        return names