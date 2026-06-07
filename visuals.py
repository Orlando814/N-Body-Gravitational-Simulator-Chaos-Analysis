from vpython import vec, color, sphere, rate, canvas, label, button, slider, menu, arrow, graph, gcurve, wtext
from Simulator import *
from Config.ConfigClass import Config
import numpy as np
from GenericConts import config_names, load_yaml

running = True
reverse = False
step = 0
bodies = []
labels = []
arrows = []
rate_value = 150
body_num = 0
scene = canvas(width=1200, height=800, title="Three Body Problem")
body_dist_1 = -1
body_dist_2 = -1
gc = gcurve(color=color.blue)
sim_history = None
body_names = []
config_changed = False
current_config = -1


def visual_loop(sim_hist: np.ndarray, radius: np.ndarray, dt: float) -> None:
    global step, rate_value, reverse, body_num, body_dist_1, body_dist_2, gc, sim_history, labels, arrows, body_names, config_changed, running, current_config
    sim_history = sim_hist
    body_num = np.size(sim_history[0, :, 0], axis=0)
    scene.background = color.black
    create_bodies_labels_arrows(sim_history, radius)
    body_names = body_name_list()
    pause_b = button(text="Pause", pos=scene.title_anchor, bind=button_event, id="Pause")
    reverse_b = button(text="Reverse", pos=scene.title_anchor, bind=button_event, id="Reverse")
    button(text="Reset", pos=scene.title_anchor, bind=button_event, id="Reset")
    drag_s = slider(bind=slider_event, pos=scene.title_anchor, min=0, max=np.size(sim_history, axis=0), step=1, id="Drag")
    rate_s = slider(bind=slider_event, pos=scene.title_anchor, min=1, max=1500, step=1, id="Rate")
    follow_m = menu(bind=menu_event, pos=scene.title_anchor, choices=body_names, id="Follow")
    dist_graph_m1 = menu(bind=menu_event, pos=scene.title_anchor, choices=body_names, id="Body 1")
    dist_graph_m2 = menu(bind=menu_event, pos=scene.title_anchor, choices=body_names, id="Body 2")
    menu(bind=menu_event, pos=scene.title_anchor, choices=config_names(), id="Current Config")
    graph(title='Distance Between Two Bodies', xtitle='Time (s)', ytitle='Distance (m)', xmin=0)
    frame_count = wtext(pos = scene.title_anchor)
    while True:
        if config_changed:
            config = Config()
            config.body = load_yaml(config_names()[current_config] + ".yaml")
            new_body = BodyConfig(config)
            state = new_body.state_array
            mass = new_body.mass
            radius = new_body.radius
            simulator = Simulator(SimConfig(), state, mass, radius)
            simulator.simulate()
            sim_history = simulator.history
            body_num = np.size(sim_history, axis=1)
            for body in bodies:
                body.clear_trail()
                body.visible = False
            for arrow in arrows:
                arrow.visible = False
            for label in labels:
                label.visible = False
            body_names = body_name_list()
            create_bodies_labels_arrows(sim_history, radius)
            reverse = False
            rate_value = 150
            body_dist_1 = -1
            body_dist_2 = -1
            gc.delete()
            running = True
            step = 0
            pause_b.text = "Pause"
            reverse_b.text = "Reverse"
            drag_s.value = 0
            rate_s.value = rate_value
            follow_m.choices = body_names
            dist_graph_m1.choices = body_names
            dist_graph_m2.choices = body_names
            scene.center = vec(0, 0, 0)
            scene.range = 1500000000
            scene.autoscale = True
            config_changed = False
        if running:
            rate(rate_value)
            frame_count.text = "Frame: " + str(step) + "/" + str(np.size(sim_history, axis = 0))
            if step == np.size(sim_history, axis=0):
                for body in bodies:
                    body.clear_trail()
                step = 0
                gc.delete()
            if step < 0 and reverse:
                step = np.size(sim_history, axis=0) - 1
                for body in bodies:
                    body.clear_trail()
                gc.delete()
            if (body_dist_1 >= 0) and (body_dist_2 >= 0):
                gc.plot(dt * step,
                        np.linalg.norm(sim_history[step, body_dist_1, 0] - sim_history[step, body_dist_2, 0]))
            count = 0
            for body in bodies:
                body.pos = vec(*sim_history[step, count, 0, :])
                count += 1
            count = 0
            for body_label in labels:
                body_label.pos = vec(*sim_history[step, count, 0, :])
                count += 1
            count = 0
            for body_arrow in arrows:
                if reverse:
                    body_arrow.axis = -vec(*sim_history[step, count, 1, :] * 350)
                else:
                    body_arrow.axis = vec(*sim_history[step, count, 1, :] * 350)
                body_arrow.pos = vec(*sim_history[step, count, 0, :])
                count += 1
            if reverse:
                step -= 1
            else:
                step += 1


def create_bodies_labels_arrows(sim_hist: np.ndarray, radius: np.ndarray):
    global body_num, bodies, labels, arrows
    bodies = []
    labels = []
    arrows = []
    for body in range(body_num):
        hue = body / body_num
        new_sphere = sphere(pos=vec(*sim_hist[0, body, 0, :]), radius=radius[body],
                            color=color.hsv_to_rgb(vec(hue, 1, 1)), make_trail=True,
                            retain=np.size(sim_hist, axis=0) * 0.05, trail_radius=radius[body] * 0.15, emissive=True)
        body_label = label(pos=vec(*sim_hist[0, body, 0, :]), xoffset=radius[body] * 0.0000005,
                           yoffset=radius[body] * 0.0000005, text="Body " + str(body + 1), box=False, line=True,
                           linecolor=color.white, opacity=0.1)
        body_arrow = arrow(pos=vec(*sim_hist[0, body, 0, :]), axis=vec(*sim_hist[0, body, 1, :] * 250),
                           color=color.hsv_to_rgb(vec(hue, 1, 1)))
        bodies.append(new_sphere)
        labels.append(body_label)
        arrows.append(body_arrow)


def button_event(evt):
    global reverse, step, running, rate_value, gc, body_dist_2, body_dist_1, bodies
    if evt.id == "Reverse":
        reverse = not reverse
        if reverse:
            evt.text = "Forward"
        else:
            evt.text = "Reverse"
        for body in bodies:
            body.clear_trail()
        gc.delete()
    elif evt.id == "Pause":
        running = not running
        if running:
            evt.text = "Pause"
        else:
            evt.text = "Run"
    elif evt.id == "Reset":
        running = True
        reverse = False
        step = 0
        rate_value = 150
        for body in bodies:
            body.clear_trail()
        body_dist_1 = -1
        body_dist_2 = -1
        gc.delete()


def slider_event(evt):
    global step, rate_value, gc, bodies
    if evt.id == "Drag":
        for body in bodies:
            body.clear_trail()
        gc.delete()
        step = evt.value
    elif evt.id == "Rate":
        rate_value = evt.value


def menu_event(evt):
    global body_dist_1, body_dist_2, body_num, sim_history, step, labels, arrows, body_names, rate_value, reverse, running, config_changed, current_config
    if evt.id == "Follow":
        if evt.index is None:
            scene.camera.follow(None)
        else:
            for body in range(body_num):
                if evt.index - 1 == body:
                    scene.camera.follow(bodies[body])
    elif evt.id == "Body 1":
        if evt.index is None:
            body_dist_1 = -1
        else:
            body_dist_1 = evt.index - 1
    elif evt.id == "Body 2":
        if evt.index is None:
            body_dist_2 = -1
        else:
            body_dist_2 = evt.index - 1
    elif evt.id == "Current Config":
        current_config = evt.index
        config_changed = True

def body_name_list():
    names = ["None"]
    for body in range(body_num):
        names.append("Body " + str(body + 1))
    return names
config = Config()
body_config = BodyConfig(config)
state = body_config.state_array
mass = body_config.mass
radius = body_config.radius
sim = SimConfig()
simulator = Simulator(sim, state, mass, radius)
simulator.simulate()
history = simulator.history
visual_loop(history, radius, simulator.dt)
