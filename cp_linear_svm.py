#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 16:27:36 2020

@author: soominkwon
"""

import numpy as np
import tensorly as tl
import functools
from scipy.optimize import minimize

class CP_LinearSVM:
    def __init__(self, CP_rank):
        """ The CP_SVM object uses a linear Support Vector Machine model with a 
            CANDECOMP/PARAFAC structure on the coefficients to solve for predictors given a
            tensor dataset. 
        
            Arguments:
                CP_rank: Rank for CP factors
        
        """
        
        self.cp_rank = CP_rank

               
    def createFactors(self, tensor):
        """ This function creates a dictionary with values being the CP factors to optimize over.
        
            Arguments:
                tensor: Tensor dataset with the sample index being the first dimension
            
            Returns:
                dictionary: Dictionary of CP factors of values being dimensions (tensor_shape x CP_rank)
        """
        
        n_dim = tensor.ndim
        dictionary = {}
        
        for i in range(1, n_dim):
            array = np.random.randn(tensor.shape[i], self.cp_rank)
            dictionary[i] = array
            
        return dictionary
    
    
    def tensorUnfold(self, tensor, mode):
        """ This function is used to unfold a tensor into a matrix given a certain mode. Please refer to
            the Kolda, Bader survey on tensor unfoldings to understand the differences of unfolding.
        
            Arguments:
                tensor: Tensor dataset for unfolding, where dimensions are (samples, tensor_shape)
                mode: Mode to unfold tensor
        """
        
        n_dim = tensor.ndim
        indices = np.arange(n_dim).tolist()
        element = indices.pop(mode)
        sample_index = indices.pop(0)
        new_indices = ([sample_index] + [element] + indices)    
        
        samples = tensor.shape[0]
        return np.transpose(tensor, new_indices).reshape(samples, tensor.shape[mode], -1)
        
 
    def getsvmFunctions(self, tensor):
        """ This function returns n functions, where n refers to the number of dimensions of the dataset
            (not including the sample dimension). For example, if the tensor dataset were of dimensions
            (samples, 10, 10, 10), this function would return a list of 3 functions to optimize each dimension
            for Linear SVM.
        
            Returns:
                optimize_functions: List of functions to optimize over
        """
        
        n_dim = tensor.ndim
        
        optimize_functions = []
        
        for i in range(1, n_dim):
            def optimizer(opt_mat, dictionary, CP_rank, C, X, Y, i=i):
                samples = X.shape[0]
     
                vec_matrix = np.reshape(opt_mat, (CP_rank*X.shape[i], 1))
                matricize_X = self.tensorUnfold(tensor=X, mode=i)
                
                temp_dict = dict(dictionary)
                del temp_dict[i]
    
                new_matrices = list(temp_dict.values())
                new_matrices = new_matrices[::-1]
    
                khatri_product = tl.tenalg.khatri_rao(new_matrices)
                X_khatri = tl.tenalg.mode_dot(matricize_X, khatri_product.T, 2)
                vec_X_khatri = np.reshape(X_khatri, (samples, CP_rank*X.shape[i]))
                Y_hat = vec_X_khatri @ vec_matrix
    
                error = (1/samples)*np.sum(np.maximum(0, (1-Y.squeeze()*Y_hat.squeeze()))) + 0.5*C*np.linalg.norm(vec_matrix)**2
    
                return error
        
            optimize_functions.append(optimizer)
            
        return optimize_functions
        

    def fit(self, tensor, labels, C=1, max_iterations=200, sensitivity=1e-6, print_iterations=True):
        iterations = 0
        error = 1
        current_error = 0
        n_dim = tensor.ndim        
        
        init_dict = self.createFactors(tensor=tensor)
        optimize_functions = self.getsvmFunctions(tensor=tensor)
        
        while (error > sensitivity) and (iterations < max_iterations):
            prev_error = current_error
            
            for i in range(1, n_dim):
                initial_guess = init_dict[i]
                solve = functools.partial(optimize_functions[i-1], dictionary=init_dict, 
                                          CP_rank=self.cp_rank, C=C, X=tensor, Y=labels)
                solver = minimize(solve, initial_guess, method='SLSQP', tol=0.0001)
                initial_guess = np.reshape(solver.x, (initial_guess.shape))
                init_dict[i] = initial_guess
                
            current_error = solver.fun
            error = abs(prev_error - current_error)
            iterations += 1
            
            if print_iterations:
                print('Current Iteration: ' + str(iterations))
                print('Current Difference in Error Value: '+ str(error) + str('\n'))
                
        return init_dict    
        
    
    def reconstructTensor(self, solved_dict, tensor_n_dim):
        """ This function takes the dictionary of CP factors that were optimized over and reconstructs
            it appropriately. 
            
            Arguments:
                solved_dict: Optimized dictionary of CP factors, output of .fit()
                tensor_dimensions: Tensor dimensions, if dimensions are (x, y, z), then tensor_dimensions=3
                
            Returns:
                recon_tensor: Recontructed tensor
                
        """
        if tensor_n_dim == 2:
            A = solved_dict[1]
            B = solved_dict[2]
            
            recon_beta = np.zeros((A.shape[0], B.shape[0]))
            
            for i in range(tensor_n_dim):
                solved_prod = np.prod(np.ix_(A[:, i], B[:, i]))
                recon_beta += solved_prod
                
            return recon_beta
                
        elif tensor_n_dim == 3:
            A = solved_dict[1]
            B = solved_dict[2]
            C = solved_dict[3]
            
            recon_beta = np.zeros((A.shape[0], B.shape[0]))
            
            for i in range(tensor_n_dim):
                solved_prod = np.prod(np.ix_(A[:, i], B[:, i], C[:, i]))
                recon_beta += solved_prod
                
            return recon_beta
    
    
    def evaluateMetrics(self, recon_tensor, true_tensor):
        """ This function computes the Mean Squared Error (MSE), Cosine Distance (or similarity), and the
            reconstruction error between the true predictor and the one evaluated by the CP algorithm.
            
            Arguments:
                recon_tensor: Reconstructed tensor
                true_tensor: True predictor to compare performance with

        """
        # normalizing tensors
        recon_tensor = recon_tensor / np.linalg.norm(recon_tensor, 'fro')
        
        # for MSE
        mse_val = np.linalg.norm(recon_tensor-true_tensor)**2
        
        # for cosine distance
        vec_recon = np.reshape(recon_tensor, (-1, 1))
        vec_true = np.reshape(true_tensor, (-1, 1))
            
        recon_norm = np.linalg.norm(vec_recon)
        true_norm = np.linalg.norm(vec_true)
        
        cos_dis = (vec_true.T @ vec_recon) / (true_norm*recon_norm)
            
        # for reconstruction error
        num = np.linalg.norm((true_tensor-recon_tensor), 'fro')
        den = np.linalg.norm(true_tensor, 'fro')
            
        recon_err = num / den
        
        #print('Squared Error:', mse_val)    
        #print('Cosine Distance:', cos_dis)
        #print('Reconstruction Error:', recon_err)
        
        return mse_val, cos_dis, recon_err
        
                    