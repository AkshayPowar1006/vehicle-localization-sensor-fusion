# -*- coding: utf-8 -*-
"""
Created on Fri May  5 11:10:33 2023

@author: Z0168020
"""


import numpy as np

sigma_a = 0.4 # process noise
sigma_m = 0.1 # measurement noise
x = np.array([72.5, 28.5, 2, 1]).T
num_states = len(x)
P = np.eye(num_states)
z = np.array([75.0, 30]).T

Q = np.array([[sigma_a, 0],
             [0, sigma_a]])
R = np.array([[sigma_m, 0],
             [0, sigma_m]])

omega = 35
alpha = 1
u = np.array([alpha])
rad = 0.5

#number of dimensions since there are 4 elements in x n sigma = 8
dim = x.shape[0]+Q.shape[0]+R.shape[0]
n_sigma = 1 + 2 * dim

x_sigma = np.zeros((x.shape[0]+Q.shape[0]+R.shape[0], n_sigma))
id1 = x.shape[0]
id2 = x.shape[0]+Q.shape[0]
x_a = np.zeros((dim,))
P_a = np.zeros((dim,dim))
x_a[:x.shape[0]] = x[:] 
P_a[:id1,:id1] = P
P_a[id1:id2, id1:id2] = Q
P_a[id2:, id2:] = R


# Now there are weights first is Wo which applied to the mean and other is Wi which is applied to all sigma points
# Wo = kappa/(kappa + dim)
# Wi = 0.5/(kappa + dim)
kappa = 3 - dim
Wo = kappa/(kappa+dim)
Wi = 0.5/(kappa+dim)


x_sigma[:,0] =  x_a
S = np.linalg.cholesky(P_a)
for i in range(1,dim+1):
    x_sigma[:,i] = x_a +  np.sqrt(dim + kappa) * S[:,i-1]
    x_sigma[:,i+dim] = x_a - np.sqrt(dim + kappa) * S[:,i-1]
    
    
def f(x,u):
    x_star = np.zeros(np.shape(x))
    for i in range(0,np.shape(x)[1]):
        x_star[0,i] = x[0,i] + x[2,i] + x[4,i]
        x_star[1,i] = x[1,i] + x[3,i] + x[5,i]
        x_star[2,i] = x[2,i] + x[4,i]
        x_star[3,i] = x[3,i] + x[5,i]

    return x_star

def mean_covariance(x_star):
    x_pred = Wo * x_star[:id1,0]
    for i in range(1, n_sigma):
        x_pred = x_pred + Wi * x_star[:id1,i]
        
    d = (x_star[:id1,0] - x_pred).reshape([-1, 1]) 
    P_pred = Wo * (d.dot(d.T))
    
    for i in range(1, n_sigma):
        d = (x_star[:id1,i] - x_pred).reshape(id1,1)
        P_pred = P_pred + Wi * (d.dot(d.T))
    
    return x_pred, P_pred

def mean_covariance2(x_corr_star):
    x_corr = Wo * x_corr_star[:,0]
    for i in range(1, n_sigma):
        x_corr = x_corr + Wi * x_corr_star[:,i]
        
    d = (x_corr_star[:,0] - x_corr).reshape([-1, 1]) 
    P_corr = Wo * (d.dot(d.T))
    
    for i in range(1, n_sigma):
        d = (x_corr_star[:,i] - x_corr).reshape(-1,1)
        P_corr = P_corr + Wi * (d.dot(d.T))
    
    return x_corr, P_corr

def cross_correlation(x_pred, x_pred_sigma, x_corr, x_corr_star):
    
    dx = (x_pred_sigma[:id1,0]-x_pred[:id1,0]).reshape([-1,1])
    dy = (x_corr_star[:,0]-x_corr).reshape([-1,1])
    
    P_pred_x_corr_x = Wo * (dx.dot(dy.T))
    
    for i in range(1, n_sigma):
            dx = (x_pred_sigma[:id1, i] - x_pred[:id1,0]).reshape([-1, 1])
            dy = (x_corr_star[:, i] - x_corr).reshape([-1, 1])
            P_pred_x_corr_x = P_pred_x_corr_x + Wi * (dx.dot(dy.T))
    
    return P_pred_x_corr_x


x_star = f(x_sigma, u)   

x_pred, P_pred = mean_covariance(x_star)


x_pred = np.vstack((x_pred, np.zeros((id2+id2-id1-id1, )))).reshape(-1,1)
# Calculating sigma points
x_pred_sigma = np.zeros((x_star.shape[0], n_sigma))
x_pred_sigma[:,0] = x_pred.reshape(-1)
P_pred_sigma = np.zeros((dim,dim))
P_pred_sigma[:id1,:id1] = P_pred
P_pred_sigma[id1:id2, id1:id2] = Q
P_pred_sigma[id2:, id2:] = R
S = np.linalg.cholesky(P_pred_sigma)

for i in range(1,dim+1):
    x_pred_sigma[:,i] = x_pred.reshape(-1) + np.sqrt(dim + kappa) * S[:,i-1]
    x_pred_sigma[:,i+dim] = x_pred.reshape(-1) - np.sqrt(dim + kappa) * S[:,i-1]
    
    
    
    
# measurement function
def h(x_pred_sigma):
    x_pred_star = np.zeros((np.shape(z)[0], n_sigma))
    for i in range(0,np.shape(x_pred_sigma)[1]):
        x_pred_star[0, i] = x_pred_sigma[0,i] + x_pred_sigma[0+id2,i]
        x_pred_star[1, i] = x_pred_sigma[1,i] + x_pred_sigma[1+id2,i]
        
    return x_pred_star


x_corr_star = h(x_pred_sigma)


x_corr, P_corr = mean_covariance2(x_corr_star)


P_pred_x_corr_x = cross_correlation(x_pred, x_pred_sigma, x_corr, x_corr_star)


K = P_pred_x_corr_x.dot(np.linalg.pinv(P_corr))
        
x_t1 = x_pred[:id1,0]+(K .dot(z - x_corr))
P_t1 = P_pred - (K.dot(P_corr.dot(K.T)))