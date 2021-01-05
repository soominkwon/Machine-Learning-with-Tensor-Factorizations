#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 00:12:15 2021

@author: soominkwon
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_data(sample_size, noise_variance):
    """ Generates data according to a model that has true CP rank of 2.
    
        Arguments:
            sample_size: Sample size
            noise_variance: Noise to add to the data
        Returns:
            beta: True predictor
            X_train: Training data
            Y_hard: Labels {-1, 1} according to X (for classification)
            Y_soft: Soft labels according to X (for linear regression)
    """
    
    # generate true beta
    A = np.array([[1]*15, [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]]).T
    B = np.array([[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], [1]*15]).T
    x_shape = A.shape[0]
    y_shape = B.shape[0]
    
    X_train = np.random.randn(sample_size, x_shape, y_shape) 
    X_train_vec = np.reshape(X_train, (sample_size, x_shape*y_shape))
    
    cross_beta = A @ B.T
    vec_cross_beta = np.reshape(cross_beta, (x_shape*y_shape, 1))
    cross_norm = np.linalg.norm(cross_beta, 'fro')
    cross_beta = cross_beta / cross_norm
    Y_soft = np.zeros((sample_size, 1))
    
    for i in range(sample_size):
        epsilon = noise_variance * np.random.randn(1, 1)
        x_i = X_train_vec[i, :]
        y_i = (x_i @ vec_cross_beta) + epsilon
        Y_soft[i, :] = y_i
        
    Y_hard = np.sign(Y_soft)
    
    return cross_beta, X_train, Y_hard, Y_soft 
    
    
if __name__ ==  "__main__":
    samples = 1000
    noise_variance = 2
    
    beta, X, Yh, Ys = generate_data(sample_size=samples, noise_variance=noise_variance)
    
    # plotting
    plt.imshow(beta, cmap='gray')
    plt.title('True Predictor')
    plt.show()
    
    