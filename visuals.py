from vpython import vec, color, sphere, rate, canvas, label, button, slider, menu, arrow, graph, gcurve
from Simulator import *
import numpy as np

running = True
reverse = False
step = 0
bodies = []
rate_value = 150
body_num = 0
scene = canvas(width=1200, height=800, title="Three Body Problem")
body_dist_1 = -1
body_dist_2 = -1
gc = gcurve(color=color.blue)


def visual_loop(sim_hist: np.ndarray, radius: np.ndarray, dt: float) -> None:
    global step, rate_value, reverse, body_num, body_dist_1, body_dist_2, gc
    body_num = np.size(sim_hist[0, :, 0], axis=0)
    scene.background = color.black
    labels, arrows = create_bodies_labels_arrows(sim_hist, radius)
    button(text="Pause", pos=scene.title_anchor, bind=button_event, id="Pause")
    button(text="Reverse", pos=scene.title_anchor, bind=button_event, id="Reverse")
    button(text="Reset", pos=scene.title_anchor, bind=button_event, id="Reset")
    slider(bind=slider_event, pos=scene.title_anchor, min=0, max=np.size(sim_hist, axis=0), step=1, id="Drag")
    slider(bind=slider_event, pos=scene.title_anchor, min=1, max=1500, step=1, id="Rate")
    menu(bind=menu_event, pos=scene.title_anchor, choices=body_name_list(), id="Follow")
    menu(bind=menu_event, pos=scene.title_anchor, choices=body_name_list(), id="Body 1")
    menu(bind=menu_event, pos=scene.title_anchor, choices=body_name_list(), id="Body 2")
    graph(title='Distance Between Two Bodies', xtitle='Time (s)', ytitle='Distance (m)', xmin=0)
    while True:
        if running:
            rate(rate_value)
            if step == np.size(sim_hist, axis=0):
                for body in bodies:
                    body.clear_trail()
                step = 0
                gc.delete()
            if step < 0 and reverse:
                step = np.size(sim_hist, axis=0) - 1
                for body in bodies:
                    body.clear_trail()
                gc.delete()
            if (body_dist_1 >= 0) and (body_dist_2 >= 0):
                gc.plot(dt * step,
                        np.linalg.norm(sim_hist[step, body_dist_1, 0] - sim_hist[step, body_dist_2, 0]))
            count = 0
            for body in bodies:
                body.pos = vec(*sim_hist[step, count, 0, :])
                count += 1
            count = 0
            for body_label in labels:
                body_label.pos = vec(*sim_hist[step, count, 0, :])
                count += 1
            count = 0
            for body_arrow in arrows:
                body_arrow.axis = vec(*sim_hist[step, count, 1, :] * 350)
                body_arrow.pos = vec(*sim_hist[step, count, 0, :])
                count += 1
            if reverse:
                step -= 1
            else:
                step += 1


def create_bodies_labels_arrows(sim_hist: np.ndarray, radius: np.ndarray):
    global body_num, bodies
    labels = []
    arrows = []
    for body in range(body_num):
        hue = body / body_num
        new_sphere = sphere(pos=vec(*sim_hist[0, body, 0, :]), radius=radius[body],
                            color=color.hsv_to_rgb(vec(hue, 1, 1)), make_trail=True, retain=200,
                            trail_radius=radius[body] * 0.1, emissive=True)
        body_label = label(pos=vec(*sim_hist[0, body, 0, :]), xoffset=radius[body] * 0.0000005,
                           yoffset=radius[body] * 0.0000005, text="Body " + str(body + 1), box=False, line=True,
                           linecolor=color.white, opacity=0.1)
        body_arrow = arrow(pos=vec(*sim_hist[0, body, 0, :]), axis=vec(*sim_hist[0, body, 1, :] * 250),
                           color=color.hsv_to_rgb(vec(hue, 1, 1)))
        bodies.append(new_sphere)
        labels.append(body_label)
        arrows.append(body_arrow)
    return labels, arrows


def button_event(evt):
    global reverse, step, running, rate_value, gc, body_dist_2, body_dist_1
    if evt.id == "Reverse":
        reverse = not reverse
        for body in bodies:
            body.clear_trail()
        gc.delete()
        if reverse:
            evt.text = "Reverse"
        else:
            evt.text = "Forward"
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
    global step, rate_value, gc
    if evt.id == "Drag":
        for body in bodies:
            body.clear_trail()
        gc.delete()
        step = evt.value
    elif evt.id == "Rate":
        rate_value = evt.value


def menu_event(evt):
    global body_dist_1, body_dist_2, body_num
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


def body_name_list():
    names = ["None"]
    for body in range(body_num):
        names.append("Body " + str(body + 1))
    return names


state = BodyConfig().state_array
mass = BodyConfig().mass
radius = BodyConfig().radius
sim = SimConfig()
simulator = Simulator(sim, state, mass, radius)
simulator.simulate()
history = simulator.history
visual_loop(history, radius, simulator.dt)
