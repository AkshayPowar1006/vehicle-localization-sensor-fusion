# -*- coding: utf-8 -*-
"""
Created on Thu May 11 16:55:34 2023

@author: Z0168020
"""

import numpy as np
import math
from math import sin, cos, pi
import matplotlib.pyplot as plt
num_states = 4
measu_inputs = 6 # position X and postion Y
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
        self.rad_w =  0.055
        self.num_sigma = (2 * self.dim_a) + 1
        self.kappa = 3 - self.dim_a
        self.alpha = 0.001
        self.beta = 2.0
        self.lamda = self.alpha**2 * (self.dim_a + self.kappa) - self.dim_a
        self.scale = np.sqrt(self.dim_a + self.kappa)
        
        # calculate unscented weights
        # self.W0m = self.W0c = self.lambda_ / (self.dim_a + self.lambda_)
        # self.W0c = self.W0c + (1.0 - alpha_2 + self.beta)
        # self.Wi = 0.5 / (self.dim_a + self.lambda_)
        
        self.Wo = self.kappa / (self.dim_a + self.kappa)
        self.Wi = 1 / (2 * (self.dim_a + self.kappa))
        
        self.X = np.zeros((self.dim_a, 1))
        self.P = np.zeros((self.dim_a, self.dim_a))
        
        self.idx1, self.idx2, self.idx3  = self.dim_state, self.dim_state + self.dim_q, self.dim_state + self.dim_q + self.dim_r
        
        self.X[:self.dim_state] = initial_state
        self.P[:self.idx1, :self.idx1] = initial_covarince
        self.P[self.idx1:self.idx2, self.idx1:self.idx2] = process_noise
        self.P[self.idx2:self.idx3, self.idx2:self.idx3] = measurement_noise
        
        #print(f'State = \n {self.X}\n')
        #print(f'Covariance = \n {self.P}\n')
        
        
        
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
        
        
        
        x_sigma_star = self.f( xx_sigmas, xq_sigmas, dt)        
        x_pred, P_xx = self.calculate_mean_covariance(x_sigma_star)
        self.U[1] = self.X[3]
        
        self.X[:self.idx1,0] = x_pred
        self.P[:self.idx1, :self.idx1] = P_xx.reshape(np.shape(self.P[:self.idx1, :self.idx1]))
        
        
        
    def update(self, z, dt: float) -> None:
        print("Update")
        
        x_sigma = self.calc_sigma_points()
        
        xx_sigmas = x_sigma[:self.idx1,:]
        xq_sigmas = x_sigma[self.idx1:self.idx2,:]
        
        y_sigma_star = self.h( xx_sigmas, xq_sigmas, dt)
        
        y_pred, P_yy = self.calculate_mean_covariance(y_sigma_star)
        
        Pzy = self.calculate_cross_covariance(self.X[:self.idx1,0], xx_sigmas, y_pred, y_sigma_star)
        
        K = Pzy.dot(np.linalg.pinv(P_yy))
                
        x_corr = self.X[:self.idx1,0].reshape(-1,1)+(K .dot(z - y_pred.reshape(-1,1)))
        P_corr = self.P[:self.idx1,:self.idx1]- (K.dot(P_yy.dot(K.T)))
        
        self.X[:self.idx1,0] = x_corr.ravel()
        self.P[:self.idx1, :self.idx1] = P_corr
        
        
        
    
    def calculate_mean_covariance(self, x_sigma_star):
        
        x = self.Wo * x_sigma_star[:,0]
        for pt in range(1, self.num_sigma):
            x = x + self.Wi * x_sigma_star[:,pt]
        
        
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
            x[0, pt] = xx_sigma[0, pt] + self.U[0]*cos(xx_sigma[2,pt])*self.rad_w*dt + xq_sigma[0,pt] + xq_sigma[1,pt]
            x[1, pt] = xx_sigma[1, pt] + self.U[0]*sin(xx_sigma[2,pt])*self.rad_w*dt + xq_sigma[0,pt] + xq_sigma[1,pt]
            x[2, pt] = xx_sigma[2, pt] + self.U[1]*dt + xq_sigma[1,pt]
            x[3, pt] = xx_sigma[3, pt] + xq_sigma[1,pt]
            
        return x
    
    def h(self,xx_sigma, xr_sigma, dt: float) -> None:
        z = np.zeros((self.Z.shape[0], self.num_sigma))
        
        for pt in range(0, self.num_sigma):
            theta = xx_sigma[2,pt]
            z[0,pt] = xx_sigma[0, pt] + cos(theta + pi) + xr_sigma[0,pt]
            z[1,pt] = xx_sigma[1, pt] + sin(theta + pi) + xr_sigma[1,pt]
            z[2,pt] = xx_sigma[0, pt] + cos(theta + pi) + xr_sigma[0,pt]
            z[3,pt] = xx_sigma[1, pt] + sin(theta + pi) + xr_sigma[1,pt]
            z[4,pt] = xx_sigma[0, pt] + cos(theta - pi) + xr_sigma[0,pt]
            z[5,pt] = xx_sigma[1, pt] + sin(theta - pi) + xr_sigma[1,pt]
            
        return z
    
    @property
    def cov(self) -> np.array:
        return self.P[:self.idx1,:self.idx1].reshape(num_states,num_states)

    @property
    def mean(self) -> np.array:
        return self.X[:self.idx1,0].reshape(num_states,1)
        
plt.ion()
plt.figure(figsize=(12,10))

        
num_states = 4
control_inputs = 2 
# define initial state consisting positionX, positionY and theta
init_state =  np.array([20,
                        50,
                        1.2,
                        0.001]).reshape(num_states,1)

#state_1_cov, state_2_cov, state_3_cov, state_4_cov = 1, 1, 0.1, 0.001
init_cov = np.array([[1,   0,   0,     0],
                     [0,   1,   0,     0],
                     [0,   0, 0.1,     0],
                     [0,   0,   0, 0.001]])

u = np.array([36.36,
               0.00001]).reshape(2,1)      


#process_variance = np.array([[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.005, 0], [0, 0, 0, 0.00005]])
process_variance = np.array([[0.5, 0], [0, 0.0005,]])


measu_variance = 0.1
Q = np.eye(control_inputs).dot(process_variance)       
R = np.eye(measu_inputs).dot(0.1) 

#measurement
z = np.array([20,50,20,50,20,50]).reshape(6,1)       
 
ukf = UKF(init_state, init_cov, z, u, Q, R) 


# minimum time step (dt) used in the simulation
time_step = 1
# total number of time steps for which simulation will run
total_time_steps = 120
 # time step at which measurement is received
measurement_rate = 1

true_mean_vehicle = np.zeros(( num_states, 1, total_time_steps))
true_omega = np.full((total_time_steps,), 0.0)

# defining the initial true mean state of a system at time t=0
true_omega[0] = 36.36

true_mean_vehicle[:,:,0] = np.array([20,50,1.2,0.001]).reshape(num_states,1)

mean_state_estimate_1 = np.zeros(( num_states, 1, total_time_steps))  #property of kalman_filter class

mean_after_prediction = np.zeros(( num_states, 1, total_time_steps))
mean_after_gps_update = np.zeros(( num_states, 1, total_time_steps))
updatesteps= []


for step in range(total_time_steps):

    if step>0:        
        
        phi = true_mean_vehicle[2,:,step-1] 
        
        true_mean_vehicle[:,:,step] = true_mean_vehicle[:,:,step-1] +  np.vstack((time_step * true_omega[step-1] * np.array([[cos(phi)],[sin(phi)]]) * rad_wheel,
                                                                                  true_mean_vehicle[3,:,step-1]*time_step,
                                                                                  0))  
        
        if step > 50 and step < 70:
            true_omega[step] = 0.97*(true_omega[step-1])
        
        if step > 50 and step < 70:
            true_mean_vehicle[3,:,step] = (true_mean_vehicle[3,:,step-1])-0.0001
        else:
            true_omega[step] = true_omega[step-1]
            true_mean_vehicle[3,0,step] = true_mean_vehicle[3,0,step-1]

            
        ukf.predict(dt=time_step)
        print(step)
    
        if step % measurement_rate == 0:
            updatesteps.append(step)
            
            # mean before update
            mean_after_prediction[:,:,step] = ukf.mean
            ''' 
            Update by adding the random Gaussian noise into the true mean position
            '''
            mgps1 = (true_mean_vehicle[0:2,:,step] + np.array([[cos(phi+pi)],[sin(phi+pi)]]) + np.random.randn() * np.sqrt(measu_variance))
            mgps2 = (true_mean_vehicle[0:2,:,step] + np.array([[cos(phi+pi/4)],[sin(phi+pi/4)]]) + np.random.randn() * np.sqrt(measu_variance))
            mgps3 = (true_mean_vehicle[0:2,:,step] + np.array([[cos(phi-pi/4)],[sin(phi-pi/4)]])  + np.random.randn() * np.sqrt(measu_variance))
            
    
            measurement_update = np.vstack((mgps1,mgps2,mgps3))            
            
            ukf.update(z=measurement_update, dt=time_step)
            
            # mean after update
            mean_after_gps_update[:,:,step] = ukf.mean

    mean_state_estimate_1[:,:,step] = ukf.mean
    
updatesteps = np.array(updatesteps)

plt.subplot(4, 1, 1)
plt.title('PositionXGPS')
plt.plot(true_mean_vehicle[0,:,updatesteps], 'ko--', label = "True")
plt.plot(mean_after_prediction[0,:,updatesteps], 'co--', label = "Pred")
plt.plot(mean_after_gps_update[0,:,updatesteps], 'ro--', label = "Corr")
#plt.plot(mean_after_prediction[:,0,0], 'c--', label = "Pred")
#plt.plot(mean_after_gps_update[:,0,0], 'r--', label = "Corr")
plt.legend(loc='lower right')

plt.subplot(4, 1, 2)
plt.title('PositionYGPS')
plt.plot(true_mean_vehicle[1,:,updatesteps], 'ko--', label = "True")
plt.plot(mean_after_prediction[1,:,updatesteps], 'co--', label = "Pred")
plt.plot(mean_after_gps_update[1,:,updatesteps], 'ro--', label = "Corr")
#plt.plot(mean_after_prediction[:,1,0], 'c--', label = "Pred")
#plt.plot(mean_after_gps_update[:,1,0], 'r--', label = "Corr")
plt.legend(loc='lower right')

plt.subplot(4, 1, 3)
plt.title('Heading')
plt.plot(true_mean_vehicle[2,:,updatesteps], 'ko--', label = "True")
plt.plot(mean_after_prediction[2,:,updatesteps], 'co--', label = "Pred")
plt.plot(mean_after_gps_update[2,:,updatesteps], 'ro--', label = "Corr")
plt.legend(loc='upper right')

plt.subplot(4, 1, 4)
plt.title('Heading Change')
plt.plot(true_mean_vehicle[3,:,updatesteps], 'ko--', label = "True")
plt.plot(mean_after_prediction[3,:,updatesteps], 'co--', label = "Pred")
plt.plot(mean_after_gps_update[3,:,updatesteps], 'ro--', label = "Corr")
plt.legend(loc='upper right')


plt.tight_layout()
plt.show()
      
        
        
        