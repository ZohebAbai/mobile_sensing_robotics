#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author : Zoheb Abai
# Repo : https://github.com/ZohebAbai/mobile_sensing_robotics/

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils
from scipy import ndimage
from sklearn.cluster import KMeans


def compute_corners(I, type):
    """
    Compute corner keypoints using Harris and Shi-Tomasi criteria
    
    Parameters
    ----------
    I : float [MxN] 
        grayscale image

    type :  string
            corner type ('harris' or 'Shi-Tomasi')
    
    Returns
    -------
    corners : numpy array [num_corners x 2] 
              Coordinates of the detected corners.    
    """
    # Detect corners 
    if type == 'harris':
        dst = cv2.cornerHarris(I, 2, 3, 0.04)
        # Dilate corner image to enhance corner points
        dst = cv2.dilate(dst, None)
        # Thresholding for an optimal value
        thresh = 0.1*dst.max()
        corners_ls = []
        for i in range(0, dst.shape[0]):
            for j in range(0, dst.shape[1]):
                if(dst[i, j] > thresh):
                    corners_ls.append([i, j])
        corners_arr = np.array(corners_ls)
        # to level up with shitomasi results we cluster to filter their centers
        kmeans = KMeans(n_clusters=25, random_state=0).fit(corners_arr)
        corners = np.array(kmeans.cluster_centers_, dtype='int64')

    if type == 'shi-tomasi':
        corners = cv2.goodFeaturesToTrack(I, 25, 0.01, 10)
        corners = np.squeeze(np.int0(corners), axis=1)

    return corners


def compute_descriptors(I, method='sift'):
    """
    Computes a 128 bit descriptor as described in the lecture.
    Parameters
    ----------
    I : float [MxN]
        grayscale image as a 2D numpy array 
    
    method : string
        key points and feature descriptors using an specific method
    
    Returns
    -------
    kps : numpy array [num_corners x 2] 
        Coordinates of the detected corners.
    features : numpy array [num_corners x 128]
        128 bit descriptors  corresponding to each corner keypoint

    """    
    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(I, None)

    return kps, features


def compute_matches(D1, D2):
    """
    Computes matches for two images using the descriptors.
    Uses the Lowe's criterea to determine the best match.

    Parameters
    ----------
    D1 : numpy array [num_corners x 128]
         descriptors for image 1 corners
    D2 : numpy array [num_corners x 128]
         descriptors for image 2 corners
 
    Returns
    ----------
    M : numpy array [num_matches x 2]
        [cornerIdx1, cornerIdx2] each row contains indices of corresponding keypoints 
    """

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(D1, D2, k=2) 
    print("Raw matches (knn):", len(matches))
    
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    return good


def plot_matches(I1, I2, C1, C2, M):
    """ 
    Plots the matches between the two images
    """
    # Create a new image with containing both images
    W = I1.shape[1] + I2.shape[1]
    H = np.max([I1.shape[0], I2.shape[0]])
    I_new = np.zeros((H, W), dtype=np.uint8)
    I_new[0:I1.shape[0], 0:I1.shape[1]] = I1
    I_new[0:I2.shape[0], I1.shape[1]:I1.shape[1] + I2.shape[1]] = I2

    # plot matches
    plt.imshow(I_new, cmap='gray')
    for i in range(M.shape[0]):
        p1 = C1[M[i, 0]].pt
        p2 = C2[M[i, 1]].pt + np.array([I1.shape[1], 0])
        plt.plot(p1[0], p1[1], 'rx')
        plt.plot(p2[0], p2[1], 'rx')
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], '-y')


def compute_homography_ransac(C1, C2, M):
    """
    Implements a RANSAC scheme to estimate the homography and the set of inliers.

    Parameters
    ----------
    C1 : numpy array [num_corners x 2]
         corner keypoints for image 1 
    C2 : numpy array [num_corners x 2]
         corner keypoints for image 2
    M  : numpy array [num_matches x 2]
        [cornerIdx1, cornerIdx2] each row contains indices of corresponding keypoints 

    Returns
    ----------
    H_final : numpy array [3 x 3]
            Homography matrix which maps in point image 1 to image 2 
    M_final : numpy array [num_inlier_matches x 2]
            [cornerIdx1, cornerIdx2] each row contains indices of inlier matches
    """

    kps1 = np.float32([kp.pt for kp in C1])
    kps2 = np.float32([kp.pt for kp in C2])

    # construct the two sets of points
    pts1 = np.float32([kps1[m.queryIdx] for m in M])
    pts2 = np.float32([kps2[m.trainIdx] for m in M])
        
    # estimate the homography between the sets of points
    (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    return H, M


# Calculate the geometric distance between estimated points and original points, namely residuals.
def compute_residual(P1, P2, H):
    """
    Compute the residual given the Homography H
    
    Parameters
    ----------
    P1: [num_points x 2]
        Points (x,y) from Image 1. The keypoint in the ith row of P1
        corresponds to the keypoint ith row of P2
    P2: [num_points x 2]
        Points (x,y) from Image 2
    H: [numpy array 3x3]
        Homography which maps P1 to P2
    
    Returns:
    ----------
    residual : float
                residual computed for the corresponding points P1 and P2 
                under the transformation given by H
    """

    # TODO: Compute residual given Homography H

    return residual


def calculate_homography_four_matches(P1, P2):
    """
    Estimate the homography given four correspondening keypoints in the two images.

    Parameters
    ----------
    P1: [num_points x 2]
        Points (x,y) from Image 1. The keypoint in the ith row of P1
        corresponds to the keypoint ith row of P2
    P2: [num_points x 2]
        Points (x,y) from Image 2

    Returns:
    ----------
    H: [numpy array 3x3]
        Homography which maps P1 to P2 based on the four corresponding points
    """

    if P1.shape[0] or P2.shape[0] != 4:
        print('Four corresponding points needed to compute Homography')
        return None

    # loop through correspondences and create assemble matrix
    # A * h = 0, where A(2n,9), h(9,1)

    A = []
    for i in range(P1.shape[0]):
        p1 = np.array([P1[i, 0], P1[i, 1], 1])
        p2 = np.array([P2[i, 0], P2[i, 1], 1])

        a2 = [
            0, 0, 0, -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2],
            p2[1] * p1[0], p2[1] * p1[1], p2[1] * p1[2]
        ]
        a1 = [
            -p2[2] * p1[0], -p2[2] * p1[1], -p2[2] * p1[2], 0, 0, 0,
            p2[0] * p1[0], p2[0] * p1[1], p2[0] * p1[2]
        ]
        A.append(a1)
        A.append(a2)

    A = np.array(A)

    # svd composition
    # the singular value is sorted in descending order
    u, s, v = np.linalg.svd(A)

    # we take the “right singular vector” (a column from V )
    # which corresponds to the smallest singular value
    H = np.reshape(v[8], (3, 3))

    # normalize and now we have H
    H = (1 / H[2, 2]) * H

    return H