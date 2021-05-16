#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Zoheb Abai
# Repo : https://github.com/ZohebAbai/mobile_sensing_robotics/


import math
import numpy as np
import matplotlib.pyplot as plt


def map2world(i,j,x_init,grid_map):
    #Converting map cells to world frame    
    origin = [grid_map.shape[0]/2,grid_map.shape[1]/2]
    new_pose = [0,0,0]
    new_pose[0] = (i - origin[0])*0.01 + x_init[0]
    new_pose[1] = (j-origin[1]) * 0.01 + x_init[1]

    return new_pose


def plot_gridmap(gridmap):
    plt.figure()
    plt.imshow(gridmap, cmap='Greys')


def inv_motion_model(u_t):
    trans = math.sqrt((u_t[1][0]-u_t[0][0])**2 + (u_t[1][1]-u_t[0][1])**2)
    rot1  = math.atan2((u_t[1][1]-u_t[0][1]),(u_t[1][0]-u_t[0][0])) - u_t[0][2]
    rot2  = u_t[1][2] - u_t[0][2] - rot1

    return rot1, trans, rot2


def prob(query, std):
    return max(0,(1/(math.sqrt(6)*std)- (abs(query)/(6 * (std**2)))))


def motion_model_odometry(x_init,x_query,u_t,alpha,gridmap = True):
    # As we are giving position and not the pose the value of p3 = 1
    if gridmap == True:

        rot1,trans, rot2 = inv_motion_model(u_t)

        rot1_hat, trans_hat , rot2_hat = inv_motion_model([x_init,x_query])

        p1 = prob(rot1 - rot1_hat,alpha[0]* abs(rot1)+ alpha[1]*trans)
        p2 = prob(trans - trans_hat,alpha[2]*trans + alpha[3] * (abs(rot1)+ abs(rot2)))
        p3 = 1

    else:
        rot1,trans, rot2 = inv_motion_model(u_t)

        rot1_hat, trans_hat , rot2_hat = inv_motion_model([x_init,x_query])

        p1 = prob(rot1 - rot1_hat,alpha[0]* abs(rot1)+ alpha[1]*trans)
        p2 = prob(trans - trans_hat,alpha[2]*trans + alpha[3] * (abs(rot1)+ abs(rot2)))
        p3 = prob(rot2-rot2_hat, alpha[0] * abs(rot2)  + alpha[1] * trans)

    return p1*p2*p3


def sample_distribution(std):
    return ((math.sqrt(6)/2)*(np.random.uniform(-std,std)+np.random.uniform(-std,std)))


def sample_motion_model(x_init,u_t,alpha):
    rot1, trans, rot2 = inv_motion_model(u_t)

    rot1_hat = rot1 + sample_distribution(alpha[0]* abs(rot1)+ alpha[1]*trans)
    trans_hat = trans + sample_distribution(alpha[2]*trans + alpha[3] * (abs(rot1)+ abs(rot2)))
    rot2_hat  = rot2 + sample_distribution(alpha[0] * abs(rot2)  + alpha[1] * trans)

    x_new = x_init[0] + trans_hat * np.cos(x_init[2]+rot1_hat)
    y_new = x_init[1] + trans_hat * np.sin(x_init[2]+rot1_hat)
    theta_new = x_init[2] + rot1_hat + rot2_hat

    return x_new,y_new,theta_new
