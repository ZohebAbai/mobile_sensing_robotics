#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


# Plot Belief
def plot_belief(belief):
    plt.figure()

    ax = plt.subplot(2, 1, 1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")

    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")


# Motion Model
def motion_model(action, belief):
    p_ff = p_bb = 0.7
    p_bf = p_fb = 0.1
    p_stay = 0.2

    belief_out = np.zeros(len(belief))

    if action == "F":

        for i in range(len(belief)):

            # Calculating the update for cell in between 0 to 14, basically one cell can be reached if F is the command
            # by either it moves forward from xcell-1 or it moves backward from xcell+1 or it stays where it is
            if i < 14 and i > 0:
                belief_out[i] = (belief[i - 1] * p_ff + belief[i + 1] * p_fb + belief[i] * p_stay)

            # for 0 given the forward update you change its belief in two cases either it is there or moves backward
            # from xcell+1
            elif i == 0:
                belief_out[i] = (belief[i + 1] * p_fb + belief[i] * p_stay)

            # for 14 given the forward update you change its belief in two cases either it is there or moves forward
            # from xcell-1
            else:
                belief_out[i] = (belief[i - 1] * p_ff + belief[i] * p_stay)

    else:
        for i in range(len(belief)):
            # Calculating the update for cell in between 0 to 14, basically one cell can be reached if B is the command
            # by either it moves backward from xcell+1 or it moves forward from xcell-1 or it stays where it is
            if i < 14 and i > 0:
                belief_out[i] = (belief[i + 1] * p_bb + belief[i - 1] * p_fb + belief[i] * p_stay)

            # for 0 given the backward update you change its belief in two cases either it is there or moves backward
            # from xcell+1
            elif i == 0:
                belief_out[i] = (belief[i + 1] * p_bb + belief[i] * p_stay)

            # for 14 given the backward update you change its belief in two cases either it is there or moves forward
            # from xcell-1
            else:
                belief_out[i] = (belief[i - 1] * p_bf + belief[i] * p_stay)

    return belief_out


# Observation/Sensor Model
def sensor_model(observation, belief, world):
    p_white = 0.7
    p_black = 0.9
    belief_out = np.copy(belief)

    if observation == 0:

        for i, val in enumerate(world):
            # Assuming black observation by sensor and world is black too
            if val == 0:
                belief_out[i] = p_black * belief[i]

            # Accounting for sensor noise in the sensor that the world is white and sensor gives black
            else:
                belief_out[i] = (1 - p_white) * belief[i]

    else:
        for i, val in enumerate(world):
            # Assuming white observation by sensor and world is white too
            if val == 1:
                belief_out[i] = p_white * belief[i]

            # Accounting for sensor noise in the sensor that the world is black and sensor gives white
            else:
                belief_out[i] = (1 - p_black) * belief[i]

    # Normalizing it to get the PDF so sum of all elements must be one
    return belief_out / sum(belief_out)


# Recursive Bayes Filter
def recursive_bayes_filter(actions, observations, belief, world):
    # Initial position observation/sensor model
    belief_sensor = sensor_model(observations[0], belief, world)

    # Recursive calculation for each action
    for i, action in enumerate(actions):
        mot_belief = motion_model(action, belief_sensor)
        belief_sensor = sensor_model(observations[i + 1], mot_belief, world)

    return belief_sensor