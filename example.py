#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 01:04:21 2021

@author: soominkwon
"""

import numpy as np
import matplotlib.pyplot as plt
from generate_data import generate_data
from cp_linear_svm import CP_LinearSVM
from cp_linear_regression import CP_LinearRegression
from cp_logistic_regression import CP_LogisticRegression
from traditional_ml import traditionalML

# initialize variables
sample_size = 1500
variance = 1.5
CP_rank = 2

# generate data
true_pred, X, Y_hard, Y_soft = generate_data(sample_size, variance)


# training CP SVM model
CP_SVM = CP_LinearSVM(CP_rank=CP_rank) # instantiating class
svm_dict = CP_SVM.fit(X, Y_hard) # fitting data (X, y)
cp_svm_reconstructed = CP_SVM.reconstructTensor(solved_dict=svm_dict, tensor_n_dim=2) # reconstruct tensor


# training traditional SVM model
vec_SVM = traditionalML(algorithm='Linear SVM') # instantiating class
vec_X = np.reshape(X, (sample_size, -1)) # vectorizing data
vec_svm_params = vec_SVM.fit(vec_X, Y_hard)
vec_svm_reconstructed = np.reshape(vec_svm_params, (cp_svm_reconstructed.shape))


# comparing CP SVM to traditional SVM model
cp_mse_val, cp_cos_dis, cp_recon_err = CP_SVM.evaluateMetrics(cp_svm_reconstructed, true_pred) # evaluate metrics
vec_mse_val, vec_cos_dis, vec_recon_err = CP_SVM.evaluateMetrics(vec_svm_reconstructed, true_pred)

print('CP MSE:', cp_mse_val)
print('Traditional MSE:', vec_mse_val, '\n')

print('CP Cosine Distance:', cp_cos_dis)
print('Traditional Cosine Distance:', vec_cos_dis, '\n')

print('CP Reconstruction Error:', cp_recon_err)
print('Traditional Reconstruction Error:', vec_recon_err, '\n')


# visualizing the solved parameters
plt.imshow(true_pred, cmap='gray')
plt.title('True Predictor')
plt.show()

plt.imshow(cp_svm_reconstructed, cmap='gray')
plt.title('CP Solved Predictor')
plt.show()

plt.imshow(vec_svm_reconstructed, cmap='gray')
plt.title('Traditional Solved Predictor')
plt.show()