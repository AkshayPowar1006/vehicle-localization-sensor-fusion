# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:28:51 2023

@author: Z0168020
"""

import numpy as np

#sigma_a = 0.5 # process noise
sigma_m = 0.9 # measurement noise
x = np.array([72.5, 28.5, 2, 1]).T
omega = 35
alpha = 0.8
u = np.array([alpha])
rad = 0.5

P = np.array([[1, 0.0, 0.0, 0.0],
              [0.0, 1, 0.0, 0.0],
              [0.0, 0.0, 1, 0.0],
              [0.0, 0.0, 0.0, 1]])
z = np.array([75.0, 30]).T
Q = np.array([[alpha*rad, 0],
             [0, alpha*rad]])
R = np.array([[sigma_m, 0],
             [0, sigma_m]])

#number of dimensions since there are 4 elements in x n sigma = 8
dim = x.shape[0]
n_sigma = 1 + 2 * dim
x_sigma = np.zeros((x.shape[0], n_sigma))


# Now there are weights first is Wo which applied to the mean and other is Wi which is applied to all sigma points
# Wo = kappa/(kappa + dim)
# Wi = 0.5/(kappa + dim)
kappa = 3 - dim
Wo = kappa/(kappa+dim)
Wi = 0.5/(kappa+dim)


x_sigma[:,0] = x
S = np.linalg.cholesky(P)
for i in range(1,dim+1):
    x_sigma[:,i] = x + np.sqrt(dim + kappa) * S[:,i-1]
    x_sigma[:,i+dim] = x - np.sqrt(dim + kappa) * S[:,i-1]
    
    
def f(x,u):
    x_star = np.zeros(np.shape(x))
    for i in range(0,np.shape(x)[1]):
        x_star[0,i] = x[0,i] + x[2,i] + Q[0,0]/2
        x_star[1,i] = x[1,i] + x[3,i] + Q[1,1]/2 
        x_star[2,i] = x[2,i] + Q[0,0]
        x_star[3,i] = x[3,i] + Q[1,1]

    return x_star

def mean_covariance(x_star, P):
    x_pred = Wo * x_star[:,0]
    for i in range(1, n_sigma):
        x_pred = x_pred + Wi * x_star[:,i]
        
    d = (x_star[:,0] - x_pred).reshape(-1,1) 
    P_pred = Wo * (d.dot(d.T))
    
    for i in range(1, n_sigma):
        d = (x_star[:,i] - x_pred).reshape(-1,1)
        P_pred = P_pred + Wi * (d.dot(d.T))
    
    return x_pred, P_pred

def cross_correlation(x_pred, x_pred_sigma, x_corr, x_corr_star):
    
    dx = (x_pred_sigma[:,0]-x_pred).reshape([-1,1])
    dy = (x_corr_star[:,0]-x_corr).reshape([-1,1])
    
    P_pred_x_corr_x = Wo * (dx.dot(dy.T))
    
    for i in range(1, dim):
            dx = (x_pred_sigma[:, i] - x_pred).reshape([-1, 1])
            dy = (x_corr_star[:, i] - x_corr).reshape([-1, 1])
            P_pred_x_corr_x = P_pred_x_corr_x + Wi * (dx.dot(dy.T))
    
    return P_pred_x_corr_x

x_star = f(x_sigma, u)   
x_pred, P_pred = mean_covariance(x_star,P)

# Calculating sigma points
x_pred_sigma = np.zeros((x.shape[0], n_sigma))
x_pred_sigma[:,0] = x_pred
S = np.linalg.cholesky(P_pred)


for i in range(1,dim+1):
    x_pred_sigma[:,i] = x_pred + np.sqrt(dim + kappa) * S[:,i-1]
    x_pred_sigma[:,i+dim] = x_pred - np.sqrt(dim + kappa) * S[:,i-1]
    
    
    
    
# measurement function
def h(x_pred):
    x_pred_star = np.zeros((np.shape(z)[0],np.shape(x_pred)[1]))
    for i in range(0,np.shape(x_pred)[1]):
        x_pred_star[0, i] = x_pred[0,i] + R[0,0]
        x_pred_star[1, i] = x_pred[1,i] + R[1,1]
        
    return x_pred_star


x_corr_star = h(x_pred_sigma)

x_corr, P_corr = mean_covariance(x_corr_star,R)


P_pred_x_corr_x = cross_correlation(x_pred, x_pred_sigma, x_corr, x_corr_star)


K = P_pred_x_corr_x.dot(np.linalg.pinv(P_corr))
        
x_t1 = x_pred + (K .dot(z - x_corr))
P_t1 = P - (K.dot(P_corr.dot(K.T)))