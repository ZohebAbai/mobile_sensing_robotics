#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Zoheb Abai
# Repo : https://github.com/ZohebAbai/mobile_sensing_robotics/

import math
import matplotlib.pyplot as plt


def plot_gridmap(gridmap):
    plt.figure()
    plt.imshow(gridmap, cmap='Greys')


def prob(query,std):
    return (1/math.sqrt(2 * math.pi * (std ** 2))) * math.exp(-0.5 * ((query ** 2)/(std ** 2)))


def landmark_obs_model(z,std,x,m):
    r = math.sqrt((m[0]-x[0])**2 + (m[1]-x[1])**2 )
    p = prob(z-r,std)

    return p