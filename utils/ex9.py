#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Zoheb Abai
# Repo : https://github.com/ZohebAbai/mobile_sensing_robotics/

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from math import sin, cos, atan2, sqrt


def plot_state(mu, S, M):
    # initialize figure
    ax = plt.gca()
    ax.set_xlim([np.min(M[:, 0]) - 2, np.max(M[:, 0]) + 2])
    ax.set_xlim([np.min(M[:, 1]) - 2, np.max(M[:, 1]) + 2])
    plt.plot(M[:, 0], M[:, 1], '^r')
    plt.title('EKF Localization')

    # visualize result
    plt.plot(mu[0], mu[1], '.b')
    plot_2dcov(mu, S)


def plot_2dcov(mu, cov):
    # covariance only in x,y
    d, v = np.linalg.eig(cov[:-1, :-1])

    # ellipse orientation
    a = np.sqrt(d[0])
    b = np.sqrt(d[1])

    # compute ellipse orientation
    if v[0, 0] == 0:
        theta = np.pi / 2
    else:
        theta = np.arctan2(v[0, 1], v[0, 0])

    # create an ellipse
    ellipse = Ellipse((mu[0], mu[1]),
                      width=a * 2,
                      height=b * 2,
                      angle=np.rad2deg(theta),
                      edgecolor='blue',
                      alpha=0.3)

    ax = plt.gca()

    return ax.add_patch(ellipse)


def wrapToPi(theta):
    while theta < -np.pi:
        theta = theta + 2 * np.pi
    while theta > np.pi:
        theta = theta - 2 * np.pi
    return theta


def inv_motion_model(u_t):
    trans = sqrt((u_t[1][0]-u_t[0][0])**2 + (u_t[1][1]-u_t[0][1])**2)
    rot1  = wrapToPi(atan2((u_t[1][1]-u_t[0][1]),(u_t[1][0]-u_t[0][0])) - u_t[0][2])
    rot2  = wrapToPi(u_t[1][2] - u_t[0][2] - rot1)

    return rot1, trans, rot2


def ekf_predict(mu, sigma, u_t, R):
    theta = mu[2][0]

    rot1,trans,rot2 = inv_motion_model(u_t)

    G_t = np.array([[1, 0, -trans * sin(theta + rot1)],
                    [0, 1,  trans * cos(theta + rot1)],
                    [0,0,1]])

    V_t = np.array([[-trans * sin(theta + rot1), cos(theta + rot1),0],
                    [trans * cos(theta + rot1), sin(theta + rot1),0],
                    [1,0,1]])
    mu_bar = mu + np.array([[trans * cos(theta + rot1)],
                            [trans * sin(theta + rot1)],
                            [rot1 + rot2]])
                        
    sigma_bar = G_t @ (sigma @ G_t.T) + V_t @ (R @ V_t.T)

    return mu_bar,sigma_bar


def ekf_correct(mu_bar, sigma_bar, z, Q, M):
    for i in range(z.shape[1]):

        j = int(z[2,i])
        lx = M[j,0]
        ly = M[j,1]
        
        q = (lx - mu_bar[0,0]) ** 2  + (ly - mu_bar[1,0]) ** 2
        dist =sqrt(q)
        
        #wrap to pi as the angle must be between -pi to pi everywhere we deal with the angle

        z_hat = np.array([[dist],
                          [wrapToPi(atan2(ly - mu_bar[1,0],lx - mu_bar[0,0]) - mu_bar[2,0])]])
       
        H_t = np.array(
                [[-(lx - mu_bar[0, 0]) / dist, -(ly - mu_bar[1, 0]) / dist, 0],
                [  (ly - mu_bar[1, 0]) / q,  -(lx - mu_bar[0, 0]) / q, -1]])

        S = H_t @ sigma_bar @ H_t.T + Q
   
        K = sigma_bar @ H_t.T @ np.linalg.inv(S)
      
        mu_bar = mu_bar + (K @ (z[:2,i].reshape(2,1) - z_hat))
        mu_bar[2,0] = wrapToPi(mu_bar[2,0]) 
        sigma_bar = (np.identity(3) - (K @ H_t)) @ sigma_bar
    
    return mu_bar,sigma_bar
