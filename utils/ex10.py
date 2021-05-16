#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Zoheb Abai
# Repo : https://github.com/ZohebAbai/mobile_sensing_robotics/

import matplotlib.pyplot as plt
import numpy as np


def world2map(pose, gridmap, map_res):
    max_y = np.size(gridmap, 0) - 1
    new_pose = np.zeros_like(pose)
    new_pose[0] = np.round(pose[0] / map_res)
    new_pose[1] = max_y - np.round(pose[1] / map_res)

    return new_pose.astype(int)

def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr


def t2v(tr):
    x = tr[0, 2]
    y = tr[1, 2]
    th = np.arctan2(tr[1, 0], tr[0, 0])
    v = np.array([x, y, th])
    return v


def ranges2points(ranges, angles):
    # rays within range
    max_range = 80
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    points = np.array([
        np.multiply(ranges[idx], np.cos(angles[idx])),
        np.multiply(ranges[idx], np.sin(angles[idx]))
    ])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, np.size(points, 1))), axis=0)
    return points_hom


def ranges2cells(r_ranges, r_angles, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges, r_angles)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # world to map
    m_points = world2map(w_points, gridmap, map_res)
    
    m_points = m_points[0:2, :]
    return m_points


def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose


def init_uniform(num_particles, img_map, map_res):
    particles = np.zeros((num_particles, 4))
    particles[:, 0] = np.random.rand(num_particles) * np.size(img_map,
                                                              1) * map_res
    particles[:, 1] = np.random.rand(num_particles) * np.size(img_map,
                                                              0) * map_res
    particles[:, 2] = np.random.rand(num_particles) * 2 * np.pi
    particles[:, 3] = 1.0
    return particles


def plot_particles(particles, img_map, map_res):

    plt.matshow(img_map, cmap="gray")
    max_y = np.size(img_map, 0) - 1
    xs = np.copy(particles[:, 0]) / map_res
    ys = max_y - np.copy(particles[:, 1]) / map_res
    plt.plot(xs, ys, '.b')
    plt.xlim(0, np.size(img_map, 1))
    plt.ylim(0, np.size(img_map, 0))
    plt.show()


def wrapToPi(theta):
    while theta < -np.pi:
        theta = theta + 2 * np.pi
    while theta > np.pi:
        theta = theta - 2 * np.pi
    return theta


def sample_distribution(std):
    sd = np.random.uniform(-std,std,12)
    return np.sum(sd)/2


def sample_motion_model(x_init, u_t, alpha):
    rot1, trans, rot2 = u_t[0],u_t[1],u_t[2]

    rot1_hat = rot1 + sample_distribution(alpha[0]* abs(rot1)+ alpha[1]*trans)
    trans_hat = trans + sample_distribution(alpha[2]*trans + alpha[3] * (abs(rot1)+ abs(rot2)))
    rot2_hat  = rot2 + sample_distribution(alpha[0] * abs(rot2)  + alpha[1] * trans)

    x_new = x_init[0] + trans_hat * np.cos(x_init[2]+rot1_hat)
    y_new = x_init[1] + trans_hat * np.sin(x_init[2]+rot1_hat)
    theta_new = wrapToPi(x_init[2] + rot1_hat + rot2_hat)

    return np.array([x_new, y_new, theta_new, 1.0])


def compute_weights(pose, obs,gridmap, map_res, lookup_table):
    weight = 1

    sensor_coordinates = ranges2cells(obs[1,:].reshape(37,1),obs[0,:].reshape(37,1),pose,gridmap,map_res).T
    
    for i in range(sensor_coordinates.shape[0]):
        
        if 0<sensor_coordinates[i,0]<lookup_table.shape[1] and 0<sensor_coordinates[i,1]<lookup_table.shape[0]:
            
            weight = weight * lookup_table[sensor_coordinates[i,1],sensor_coordinates[i,0]]
        
        else:
            weight = 1e-300
    
    return weight


def resample(weights, particles):
    resampled_particles = np.zeros_like(particles) 
    J = particles.shape[0]
    r = np.random.uniform(0,1.0/J)
    c = weights[0]
    i = 0
    
    for j in range(1,J+1):
        U = r + ((j-1) * (1.0/J))
        
        while U > c:
            i = ((i + 1) % particles.shape[0])
            c = c + weights[i]
       
        resampled_particles[j-1] = particles[i]
    
    return resampled_particles
