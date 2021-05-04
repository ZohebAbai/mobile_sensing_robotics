#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import bresenham as bh

def plot_gridmap(gridmap):
    gridmap = np.array(gridmap, dtype=np.float64)
    plt.figure()
    plt.imshow(gridmap, cmap='Greys',vmin=0, vmax=1)
    
def init_gridmap(size, res):
    gridmap = np.zeros([int(np.ceil(size/res)), int(np.ceil(size/res))])
    return gridmap

def world2map(pose, gridmap, map_res):
    origin = np.array(gridmap.shape)/2
    new_pose = np.zeros((pose.shape))
    new_pose[0:] = np.round(pose[0:]/map_res) + origin[0]
    new_pose[1:] = np.round(pose[1:]/map_res) + origin[1]
    return new_pose.astype(int)

def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    tr = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return tr    

def ranges2points(ranges):
    # laser properties
    start_angle = -1.5708
    angular_res = 0.0087270
    max_range = 30
    # rays within range
    num_beams = ranges.shape[0]
    idx = (ranges < max_range) & (ranges > 0)
    # 2D points
    angles = np.linspace(start_angle, start_angle + (num_beams*angular_res), num_beams)[idx]
    points = np.array([np.multiply(ranges[idx], np.cos(angles)), np.multiply(ranges[idx], np.sin(angles))])
    # homogeneous points
    points_hom = np.append(points, np.ones((1, points.shape[1])), axis=0)
    return points_hom

def ranges2cells(r_ranges, w_pose, gridmap, map_res):
    # ranges to points
    r_points = ranges2points(r_ranges)
    w_P = v2t(w_pose)
    w_points = np.matmul(w_P, r_points)
    # covert to map frame
    m_points = world2map(w_points, gridmap, map_res)
    m_points = m_points[0:2,:]
    return m_points

def poses2cells(w_pose, gridmap, map_res):
    # covert to map frame
    m_pose = world2map(w_pose, gridmap, map_res)
    return m_pose  

def bresenham(x0, y0, x1, y1):
    l = np.array(list(bh.bresenham(x0, y0, x1, y1)))
    return l
    
def prob2logodds(p):
    logodds = np.log(p / (1 - p))
    return logodds
    
def logodds2prob(l):
    l = np.array(l, dtype=np.float128)
    prob = 1 - (1 / (1 + np.exp(l)))
    return prob
    
def inv_sensor_model(cell, endpoint, prob_occ, prob_free):
    line = bresenham(cell[0], cell[1], endpoint[0], endpoint[1])
    prob_values = []

    for i in range(len(line) - 1):
        prob_values.append(prob_free)

    prob_values.append(prob_occ)
    prob_values = np.array(prob_values).reshape((len(line), 1))
    inv_sensor_model = np.hstack((line, prob_values))

    return inv_sensor_model

def grid_mapping_with_known_poses(ranges_raw, poses_raw, occ_gridmap, map_res, prob_occ, prob_free, prior):
    # Known Poses for the grid mapping
    poses = poses2cells(poses_raw, occ_gridmap, map_res)

    # Converting cell value to the log value
    occ_gridmap = prob2logodds(occ_gridmap)

    # Given Sensor range value for every pose
    for i in range(poses.shape[0]):
        ranges = ranges2cells(ranges_raw[i], poses_raw[i], occ_gridmap, map_res).transpose()

        # update the probability within the senor range.
        for j in range(ranges.shape[0]):
            inv_sensor_val = inv_sensor_model(poses[i], ranges[j], prob_occ, prob_free)

            # Update the cell
            for k in range(len(inv_sensor_val)):
                cell = np.array([[int(inv_sensor_val[k][0]), int(inv_sensor_val[k][1])]])

                # update the grid map by converting probiblity output from the sensor to logvalue and add it to grid value.
                occ_gridmap[cell[0][0]][cell[0][1]] = occ_gridmap[cell[0][0]][cell[0][1]] + prob2logodds(
                    inv_sensor_val[k][2]) - prob2logodds(prior)

    # The cell value are converted back probability.
    occ_gridmap = logodds2prob(occ_gridmap)

    return occ_gridmap
