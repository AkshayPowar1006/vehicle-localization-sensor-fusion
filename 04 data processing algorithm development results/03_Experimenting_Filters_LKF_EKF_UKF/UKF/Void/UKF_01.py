# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:50:24 2023

@author: Z0168020
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import sys


class UKF(object):
    def __init__(self, dim_x, dim_z, Q, R, kappa=0.0):
        
        '''
        UKF class constructor
        inputs:
            dim_x : state vector x dimension
            dim_z : measurement vector z dimension
        
        - step 1: setting dimensions
        - step 2: setting number of sigma points to be generated
        - step 3: setting scaling parameters
        - step 4: calculate scaling coefficient for selecting sigma points
        - step 5: calculate weights
        '''
                
        # setting dimensions
        self.dim_x = dim_x         # state dimension
        self.dim_z = dim_z         # measurement dimension
        self.dim_v = np.shape(Q)[0]
        self.dim_n = np.shape(R)[0]
        self.dim_a = self.dim_x + self.dim_v + self.dim_n # assuming noise dimension is same as x dimension
        
        # setting number of sigma points to be generated
        self.n_sigma = (2 * self.dim_a) + 1
        
        # setting scaling parameters
        self.kappa = 3 - self.dim_a #kappa
        self.alpha = 0.001
        self.beta = 2.0

        alpha_2 = self.alpha**2
        self.lambda_ = alpha_2 * (self.dim_a + self.kappa) - self.dim_a
        
        # setting scale coefficient for selecting sigma points
        # self.sigma_scale = np.sqrt(self.dim_a + self.lambda_)
        self.sigma_scale = np.sqrt(self.dim_a + self.kappa)
        
        # calculate unscented weights
        # self.W0m = self.W0c = self.lambda_ / (self.dim_a + self.lambda_)
        # self.W0c = self.W0c + (1.0 - alpha_2 + self.beta)
        # self.Wi = 0.5 / (self.dim_a + self.lambda_)
        
        self.W0 = self.kappa / (self.dim_a + self.kappa)
        self.Wi = 0.5 / (self.dim_a + self.kappa)
        
        # initializing augmented state x_a and augmented covariance P_a
        self.x_a = np.zeros((self.dim_a, ))
        self.P_a = np.zeros((self.dim_a, self.dim_a))
        
        self.idx1, self.idx2 = self.dim_x, self.dim_x + self.dim_v
        
        self.P_a[self.idx1:self.idx2, self.idx1:self.idx2] = Q
        self.P_a[self.idx2:, self.idx2:] = R
        
        print(f'P_a = \n{self.P_a}\n')
            
    def predict(self, f, x, P):       
        self.x_a[:self.dim_x] = x
        self.P_a[:self.dim_x, :self.dim_x] = P
        
        xa_sigmas = self.sigma_points(self.x_a, self.P_a)
        
        xx_sigmas = xa_sigmas[:self.dim_x, :]
        xv_sigmas = xa_sigmas[self.idx1:self.idx2, :]
        
        y_sigmas = np.zeros((self.dim_x, self.n_sigma))              
        for i in range(self.n_sigma):
            y_sigmas[:, i] = f(xx_sigmas[:, i], xv_sigmas[:, i])
        
        y, Pyy = self.calculate_mean_and_covariance(y_sigmas)
        
        self.x_a[:self.dim_x] = y
        self.P_a[:self.dim_x, :self.dim_x] = Pyy
               
        return y, Pyy, xx_sigmas
        
    def correct(self, h, x, P, z):
        self.x_a[:self.dim_x] = x
        self.P_a[:self.dim_x, :self.dim_x] = P
        
        xa_sigmas = self.sigma_points(self.x_a, self.P_a)
        
        xx_sigmas = xa_sigmas[:self.dim_x, :]
        xn_sigmas = xa_sigmas[self.idx2:, :]
        
        y_sigmas = np.zeros((self.dim_z, self.n_sigma))
        for i in range(self.n_sigma):
            y_sigmas[:, i] = h(xx_sigmas[:, i], xn_sigmas[:, i])
            
        y, Pyy = self.calculate_mean_and_covariance(y_sigmas)
                
        Pxy = self.calculate_cross_correlation(x, xx_sigmas, y, y_sigmas)

        K = Pxy @ np.linalg.pinv(Pyy)
        
        x = x + (K @ (z - y))
        P = P - (K @ Pyy @ K.T)
        
        return x, P, xx_sigmas
        
    
    def sigma_points(self, x, P):
        
        '''
        generating sigma points matrix x_sigma given mean 'x' and covariance 'P'
        '''
        
        nx = np.shape(x)[0]
        
        x_sigma = np.zeros((nx, self.n_sigma))       
        x_sigma[:, 0] = x
        
        S = np.linalg.cholesky(P)
        
        for i in range(nx):
            x_sigma[:, i + 1]      = x + (self.sigma_scale * S[:, i])
            x_sigma[:, i + nx + 1] = x - (self.sigma_scale * S[:, i])
            
        return x_sigma
    
    
    def calculate_mean_and_covariance(self, y_sigmas):
        ydim = np.shape(y_sigmas)[0]
        
        # mean calculation
        y = self.W0 * y_sigmas[:, 0]
        for i in range(1, self.n_sigma):
            y += self.Wi * y_sigmas[:, i]
            
        # covariance calculation
        d = (y_sigmas[:, 0] - y).reshape([-1, 1])
        Pyy = self.W0 * (d @ d.T)
        for i in range(1, self.n_sigma):
            d = (y_sigmas[:, i] - y).reshape([-1, 1])
            Pyy += self.Wi * (d @ d.T)
    
        return y, Pyy
    
    def calculate_cross_correlation(self, x, x_sigmas, y, y_sigmas):
        # xdim = np.shape(x)[0]
        # ydim = np.shape(y)[0]
        
        n_sigmas = np.shape(x_sigmas)[1]
    
        dx = (x_sigmas[:, 0] - x).reshape([-1, 1])
        dy = (y_sigmas[:, 0] - y).reshape([-1, 1])
        Pxy = self.W0 * (dx @ dy.T)
        for i in range(1, n_sigmas):
            dx = (x_sigmas[:, i] - x).reshape([-1, 1])
            dy = (y_sigmas[:, i] - y).reshape([-1, 1])
            Pxy += self.Wi * (dx @ dy.T)
    
        return Pxy
    
    
def f_2(x, nu):
    xo = np.zeros((np.shape(x)[0],))
    xo[0] = x[0] + x[2] + nu[0]
    xo[1] = x[1] + x[3] + nu[1]
    xo[2] = x[2] + nu[0]
    xo[3] = x[3] + nu[1]
    return xo


class RangeMeasurement:
    def __init__(self, position):
        self.range = np.sqrt(position[0]**2 + position[1]**2)
        self.bearing = np.arctan(position[1] / (position[0] + sys.float_info.epsilon))
        self.position = np.array([position[0],position[1]])

    def actual_position(self):
        return self.position
        
    def asArray(self):
        return np.array([self.range,self.bearing])
        
    def show(self):
        print(f'range: {self.range}')
        print(f'bearing: {self.bearing}')
        
#measurement = RangeMeasurement((10.7, 5.6)) # ground-truth
#measurement.show()


def h_2(x, n):
    '''
    nonlinear measurement model for range sensor
    x : input state vector [2 x 1] ([0]: p_x, [1]: p_y)
    z : output measurement vector [2 x 1] ([0]: range, [1]: bearing )
    '''
    z = np.zeros((2,))
    #z[0] = np.sqrt((x[0] + n[0])**2 + (x[1] + n[1])**2)
    #z[1] = np.arctan((x[1] + n[1]) / ((x[0] + n[0]) + sys.float_info.epsilon))
    z[0] = x[0] + n[0]
    z[1] = x[1] + n[1]
    return z


# generate trajectory

trajectory = [[75.0, 30.0], [78.5, 32.5], [83.0, 36.0]]#, [1113.0, 611.0], [1114.0, 611.5], [1115.0, 711.0], [1116.0, 711.0]]
trajectory = np.asarray(trajectory)

traj_xlist = trajectory[:, 0]
traj_ylist = trajectory[:, 1]

print(traj_xlist)
print(traj_ylist)

# measurements = []
# for pose in trajectory:
#     meas = RangeMeasurement(pose)
#     measurements.append(meas.asArray())

measurements = np.asarray(trajectory)

x0 = np.array([72.5, 28.5, 2., 1.])

P0 = np.array([[1, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])

Q = np.array([[0.4, 0.0],
              [0.0, 0.4]])

R = np.array([[0.1, 0.0],[0.0, 0.1]])

nx = np.shape(x0)[0]
nz = np.shape(R)[0]
nv = np.shape(x0)[0]
nn = np.shape(R)[0]

ukf = UKF(dim_x=nx, dim_z=nz, Q=Q, R=R, kappa=(3 - nx))

#viewer = create_viewer('Tracking Target Trajectory', 'x (m)', 'y (m)', xlim=(8,20), ylim=(4,8))
#viewer.scatter(traj_xlist, traj_ylist, s=80, marker='o', c='blue', label='actual target poses')

x, P = x0, P0

estimates = []

for iteration, z in enumerate(measurements):
    x, P, _ = ukf.predict(f_2, x, P)
    z[0] = 75.0
    z[1] = 30.0
    x, P, _ = ukf.correct(h_2, x, P, z)
#    visualize_estimate(viewer, f'', 'g', x, P)
    
    estimates.append(x)

estimates = np.asarray(estimates).T
print(estimates)

estimates_px = estimates[0, :]
estimates_py = estimates[1, :]

estimates_vx = estimates[2, :]
estimates_vy = estimates[3, :]

#viewer.plot(estimates_px, estimates_py)

#update_plotter()







