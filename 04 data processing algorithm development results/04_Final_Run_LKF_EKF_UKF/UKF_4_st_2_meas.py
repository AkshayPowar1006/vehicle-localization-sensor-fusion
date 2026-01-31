# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:07:06 2023

@author: Z0168020
"""

import numpy as np
from math import sin, cos, pi
num_states = 4
measu_inputs = 2 # position X and postion Y
control_input = 2   
rad_wheel = 0.055


class UKF:
    
    def __init__(self, initial_state,
                 initial_covarince,
                 measurement,
                 control_input,
                 process_noise,
                 measurement_noise)-> None:
        
        self.dim_state = len(initial_state)
        self.dim_measu = len(measurement)
        self.dim_q = process_noise.shape[0]
        self.dim_r = measurement_noise.shape[0]
        self.dim_a = self.dim_state + self.dim_q + self.dim_r
        
        self.U = control_input
        self.Z = measurement
        self.Q = process_noise
        self.rad_w =  0.055
        self.num_sigma = (2 * self.dim_a) + 1
        self.kappa = 3 - self.dim_a
        self.alpha = 0.001
        self.beta = 2.0
        self.lamda = self.alpha**2 * (self.dim_a + self.kappa) - self.dim_a
        self.scale = np.sqrt(self.dim_a + self.kappa)
        
        self.Wo = self.kappa / (self.dim_a + self.kappa)
        self.Wi = 1 / (2 * (self.dim_a + self.kappa))
        
        self.X = np.zeros((self.dim_a, 1))
        self.P = np.zeros((self.dim_a, self.dim_a))
        
        self.idx1, self.idx2, self.idx3  = self.dim_state, self.dim_state + self.dim_q, self.dim_state + self.dim_q + self.dim_r
        
        self.X[:self.dim_state] = initial_state
        self.P[:self.idx1, :self.idx1] = initial_covarince
        self.P[self.idx1:self.idx2, self.idx1:self.idx2] = process_noise
        self.P[self.idx2:self.idx3, self.idx2:self.idx3] = measurement_noise
        
    def calc_sigma_points(self):
        
        x_sigma = np.zeros((len(self.X), self.num_sigma))
        x_sigma[:,0] = self.X.ravel()
        
        eigenvalues = np.linalg.eigvals(self.P)
        if np.all(eigenvalues > 0):
            print(" ")
        else:
            print("Not all eigenvalues are positive.")
        
        S  = np.linalg.cholesky(self.P)
 
        
        for i in range(len(self.X)):
            x_sigma[:, i + 1]      = self.X.ravel() + (self.scale * S[:, i])
            x_sigma[:, i + len(self.X) + 1] = self.X.ravel() - (self.scale * S[:, i])
            
        return x_sigma
    
    
    def predict(self, dt: float) -> None:
        print("Prediction")
               
        x_sigma = self.calc_sigma_points()
        
        xx_sigmas = x_sigma[:self.idx1,:]
        xq_sigmas = x_sigma[self.idx1:self.idx2,:]
        
        self.U[1] = self.X[3]
        
        x_sigma_star = self.f( xx_sigmas, xq_sigmas, dt)        
        x_pred, P_xx = self.calculate_mean_covariance(x_sigma_star)
        
        self.X[:self.idx1,0] = x_pred
        self.P[:self.idx1, :self.idx1] = P_xx.reshape(np.shape(self.P[:self.idx1, :self.idx1]))
        
        
        
    def update(self, z, flag) -> None:
        print("Update")
        
        x_sigma = self.calc_sigma_points()
        
        xx_sigmas = x_sigma[:self.idx1,:]
        xr_sigmas = x_sigma[self.idx2:self.idx3,:]
        
        if flag == 0:
            a, angle = 0.2, pi/2
        if flag == 1:
            a, angle = 1, pi
        if flag == 2:
            a, angle = 1, pi/4
        if flag == 3:
            a, angle = 1, -pi/4
            
        
        
        y_sigmas = self.h(xx_sigmas, xr_sigmas, a, angle)
        y_pred, P_yy = self.calculate_mean_covariance(y_sigmas)
        
        Pzy = self.calculate_cross_covariance(self.X[:self.idx1,0], xx_sigmas, y_pred, y_sigmas)
        
        K = Pzy.dot(np.linalg.pinv(P_yy))
                
        x_corr = self.X[:self.idx1]+(K .dot(z - y_pred.reshape(-1,1)))
        P_corr = self.P[:self.idx1,:self.idx1]- (K.dot(P_yy.dot(K.T)))
        
        self.X[:self.idx1,0] = x_corr.ravel()
        self.P[:self.idx1, :self.idx1] = P_corr
    
    def calculate_mean_covariance(self, x_sigma_star):
        
        x = self.Wo * x_sigma_star[:,0]
        for pt in range(1, self.num_sigma):
            x = x + self.Wi * x_sigma_star[:,pt]
        x.reshape(-1,1)
        
        diff = (x_sigma_star[:,0] - x).reshape(-1,1)
        Pxx = self.Wo * (diff.dot(diff.T))
        for pt in range(1, self.num_sigma):
            diff = (x_sigma_star[:,pt] - x).reshape(-1,1)
            Pxx = Pxx + self.Wi * (diff.dot(diff.T))
        
        return x, Pxx
        
    def calculate_cross_covariance(self, x, x_sigma, y, y_sigma):
        
        diff_x = (x_sigma[:,0] - x).reshape(-1,1)
        diff_y = (y_sigma[:,0] - y).reshape(-1,1)
        Pxy = self.Wo * (diff_x.dot(diff_y.T))
        for pt in range(1,self.num_sigma):
            diff_x = (x_sigma[:,pt] - x).reshape(-1,1)
            diff_y = (y_sigma[:,pt] - y).reshape(-1,1)
            Pxy = Pxy + self.Wi * (diff_x.dot(diff_y.T))
               
        return Pxy
        
        
        
    def f(self, xx_sigma, xq_sigma, dt: float) -> None:
        x = np.zeros(np.shape(xx_sigma))
        
        for pt in range(0, self.num_sigma):
            x[0, pt] = xx_sigma[0, pt] + self.U[0]*cos(xx_sigma[2,pt])*self.rad_w*dt + xq_sigma[0,pt] 
            x[1, pt] = xx_sigma[1, pt] + self.U[0]*sin(xx_sigma[2,pt])*self.rad_w*dt + xq_sigma[1,pt] 
            x[2, pt] = xx_sigma[2, pt] + self.U[1]*dt + xq_sigma[2,pt]
            x[3, pt] = xx_sigma[3, pt] + xq_sigma[3,pt]            
            
        return x
    
    
    def h(self,xx_sigma, xr_sigma, a: float, angle: float) -> None:
        z = np.zeros((self.Z.shape[0], self.num_sigma))
        
        for pt in range(0, self.num_sigma):
            #theta = xx_sigma[2,pt]
            theta = self.X[2]
            z[0,pt] = xx_sigma[0, pt] + a * cos(theta + angle) + xr_sigma[0,pt]
            z[1,pt] = xx_sigma[1, pt] + a * sin(theta + angle) + xr_sigma[1,pt]
            
        return z
    
    @property
    def cov(self) -> np.array:
        return self.P[:self.idx1,:self.idx1].reshape(num_states,num_states)

    @property
    def mean(self) -> np.array:
        return self.X[:self.idx1,0].reshape(num_states,1)
        