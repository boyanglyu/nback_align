#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm
from scipy.linalg import logm
from scipy.linalg import expm
from numpy import linalg as LA

np.set_printoptions(threshold=1000000)
import ot

import statsmodels.stats.correlation_tools

def is_pos_def(x):
    if np.all(np.linalg.eigvals(x) > 0):
        return True
    else:
        print(np.linalg.eigvals(x))
        return False

# this can be used as distance between power spectral density, a and b are power spectral density
# def hellinger_1(a, b):
#     add_part = np.trace(a + b)
#     mul_part = 2 * np.trace(sqrtm(a) @ sqrtm(b))
    
#     if add_part - mul_part < 0:
#         return 0
#     return np.sqrt(add_part - mul_part)


# def hellinger_2(a, b):
#     add_part = np.trace(a + b)
#     mul_part = 2 * np.trace(sqrtm(a@b)) # np.sqrtm(np.trace(a@b))
#     if add_part - mul_part < 0:
#         return 0
#     return np.sqrt(add_part - mul_part)


def hellinger_3(a, b):
    add_part = np.trace(a + b)
    mul_part = 2 * np.trace(sqrtm(a) @ 
                            sqrtm((inv(sqrtm(a))@b@inv(sqrtm(a))))@sqrtm(a))
    if add_part - mul_part < 0:
        return 0
    return np.sqrt(add_part - mul_part)


# def hellinger_4(a, b):
#     add_part = np.trace(a + b)
#     mul_part = 2 * np.trace(expm((logm(a) + logm(b)) / 2))
#     if add_part - mul_part < 0:
#         return 0
#     return np.sqrt(add_part - mul_part)


# def riemannian_distance(a, b):
#     eig_val = LA.eigvals(a@inv(b))
#     res = np.sum(np.log(eig_val)**2)
#     return res


def cal_mean(x, y):
    mean_x = np.mean(x,axis=1, keepdims = True)
    mean_y = np.mean(y,axis=1, keepdims = True)
    mean_diff = mean_x - mean_y

    return np.sqrt(np.sum(np.square(mean_diff)))
 

# cal_norm_mean or cal_mean
def distance_mean_mat(data, window):

    x = data
    y = data
    nSamp = int(data.shape[1]/window)
    print(nSamp)
    mean_mat = np.zeros((nSamp,nSamp))
    for i in range(nSamp):
        for j in range(nSamp):
            mean =  cal_mean(x[:, window*i: window*(i+1)],
                  y[:, window*j: window*(j+1)])  
            mean_mat[i,j] = mean 
    return mean_mat

def cal_cov(x, y, method, threshold):
    assert type(method) is int
    
    # compute covariance matrix of x and y
    cov_x = np.cov(x) 
    cov_y = np.cov(y) 
    if not is_pos_def(cov_x):
        cov_x = statsmodels.stats.correlation_tools.cov_nearest(cov_x,threshold=threshold)
        print('x need to be approx')
    if not is_pos_def(cov_y):
        cov_y = statsmodels.stats.correlation_tools.cov_nearest(cov_y,threshold=threshold)
        print('y need to be approx')
    assert is_pos_def(cov_x) 
    assert is_pos_def(cov_y) 
    
    if method == 1:
        distance = hellinger_1(cov_x, cov_y)
    elif method == 2:
        distance = hellinger_2(cov_x, cov_y)
    elif method == 3:
        distance = hellinger_3(cov_x, cov_y)
    elif method == 4:
        distance = hellinger_4(cov_x, cov_y)
    elif method == 5:
        distance = riemannian_distance(cov_x, cov_y)
    return np.linalg.norm(distance)
    

def distance_cov_mat(data, window, method, threshold):

    x = data
    y = data

    nSamp = int(data.shape[1]/window)
    print(nSamp)
    #dis_mat = np.zeros((nSamp,nSamp))
    cov_mat = np.zeros((nSamp,nSamp))
    for i in range(nSamp):
        for j in range(nSamp):
            cov =  cal_cov(x[:, window*i: window*(i+1)],
                  y[:, window*j: window*(j+1)], method, threshold)  
            cov_mat[i,j] = cov 
    
    return cov_mat


