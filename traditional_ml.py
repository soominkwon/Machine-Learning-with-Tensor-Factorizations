#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:56:38 2021

@author: soominkwon
"""

import numpy as np
import functools
from scipy.optimize import minimize

class traditionalML:
    def __init__(self, algorithm):
        """ This object allows the use of traditional ML algorithms (regularized Linear SVM, 
            Logistic Regression, and Linear Regression).
        
            Arguments:
                algorithm: Specify the algorithm to use (Linear SVM, Logistic Regression, Linear Regression)
        """
        
        if algorithm not in ['Linear SVM', 'Logistic Regression', 'Linear Regression']:
            print('ERROR: Algorithms available are [Linear SVM, Logistic Regression, Linear Regression]')
            return
        
        self.algorithm = algorithm
        
    
    def svmSolver(self, beta, C, vec_X, Y):
        """ Creates the objective function for Linear SVM.
            
            Arguments:
                beta: Vectorized predictors (weights) to solve for
                vec_X: Vectorized training data with dimensions (samples, dimensions)
                Y: Labels with dimensions (samples, 1)
             
            Returns:
                error: Objective function for linear SVM
        """
    
        samples = vec_X.shape[0]
        Y_hat = vec_X @ beta 
        error = (1/samples)*np.sum(np.maximum(0, (1-Y.squeeze()*Y_hat.squeeze()))) + 0.5*C*np.linalg.norm(beta)**2
        
        return error
    
    
    def lrSolver(self, beta, C, vec_X, Y):
        """ Creates the objective function for Logistic Regression.
            
            Arguments:
                beta: Predictors (weights) to solve for
                vec_X: Vectorized training data with dimensions (samples, dimensions)
                Y: Labels with dimensions (samples, 1)
             
            Returns:
                error: Objective function for Logistic Regression
        """    
        
        samples = vec_X.shape[0]
        Y_hat = vec_X @ beta
        error = (1/samples)*np.sum(np.log(1+np.exp(-Y_hat*Y))) + C*np.linalg.norm(beta)**2
        
        return error
    
    
    def linregSolver(self, beta, C, vec_X, Y):
        """ Creates the objective function for Logistic Regression.
            
            Arguments:
                beta: Predictors (weights) to solve for
                vec_X: Vectorized training data with dimensions (samples, dimensions)
                Y: Labels with dimensions (samples, 1)
             
            Returns:
                error: Objective function for Logistic Regression
        """    
        
        samples = vec_X.shape[0]
        Y_hat = vec_X @ beta
        error = (1/samples)*np.sum((Y.squeeze() - Y_hat.squeeze())**2)
        
        return error
    
    
    def fit(self, vec_X, Y, C=1, max_iterations=200, sensitivity=1e-6, print_iterations=True):
        """ Function to train data.
            
            Arguments:
                vec_X: Vectorized training data
                Y: Labels according to X
                C: Hyperparameter for regularization
                max_iterations: Maximum number of iterations for optimization
                sensitivity: Sensitivity of the function
                print_iterations: Prints the objective function error while training if True
                
            Returns:
                opt_weights: Optimized weights (predictors) for chosen algorithm according to (X, y)
        """
        
        if self.algorithm == 'Linear SVM':
            solver = self.svmSolver
        elif self.algorithm == 'Logistic Regression':
            solver = self.lrSolver
        elif self.algorithm == 'Linear Regression':
            solver = self.linregSolver
            
        init_beta = np.random.randn(vec_X.shape[1], 1)
        partial_solver = functools.partial(solver, C=C, vec_X=vec_X, Y=Y)
        solved = minimize(partial_solver, init_beta, method='SLSQP')
        opt_weights = solved.x
        
        return opt_weights 
    