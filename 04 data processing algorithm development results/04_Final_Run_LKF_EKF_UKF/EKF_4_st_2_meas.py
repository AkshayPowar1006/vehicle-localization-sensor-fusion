# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:09:47 2023

@author: Z0168020
"""

# Import necessary libraries
import numpy as np
from math import sin, cos, pi


num_states = 4
measu_inputs = 6
control_inputs = 2
rad_wheel =  0.055

class EKF:

    def __init__(self, initial_state,
                 initial_covariance,
                 measurement,
                 control_input,
                 process_noise,
                 measurement_noise) -> None:
 
        # mean state matrix of Gaussian Random Variable (GRV)
        # initialized as 4x1 matrix [pos X, pos Y, theta, theta_dot]
        self._x = initial_state.reshape(num_states,1)
        # covariance of state Gaussian Random Variable (GRV)
        # initialized as 4x4 identity matrix
        self._P = initial_covariance  
        # control input matrix
        # initialized as 2x1 matrix [omega, theta dot]
        self._u = control_input
        # process noise matrix
        # initialized as 4x4 matrix diag([sigma(posX)2, sigma(posY)2, sigma(theta)2, sigma(theta_dot)2])
        self._Q = process_noise
        # measurement noise matrix
        # initialized as 6x6 matrix diag([sigma(posX1)2, sigma(posY1)2, sigma(posX2)2, sigma(posY2)2, sigma(posX3)2, sigma(posY3)2])
        self._R = measurement_noise
        # measurement matrix
        # initialized as 6x1 empty matrix
        self._z = measurement  
        

    def predict(self, dt: float) -> None:
        """     
        mean prediction
            x = f(x,u) = fx + bu
        covariance prediction
            P = F P Ft + Q
        """
        
        def getJacobofF(self,dt:float):    
            theta = self._x[2]  #+ self._x[3]*dt
            F = np.array([[1,0, self._u[0]*(-sin(theta))*rad_wheel*dt, self._u[0]*(-sin(theta))*rad_wheel*dt*dt],
                          [0, 1, self._u[0]*(cos(theta))*rad_wheel*dt, self._u[0]*( cos(theta))*rad_wheel*dt*dt],
                          [0, 0, 1, dt],
                          [0, 0, 0, 1]],dtype=float)
            
            return F
        
        theta = self._x[2] #+ self._x[3]*dt
        self._u[1] =  self._x[3]
        # 4x4 State transition matrix
        f = np.eye(4)
        
        # 4x2 Control Jacobian w.r.t. omega and theta_dot
        b = np.array([[cos(theta)*rad_wheel * dt, 0],
                      [sin(theta)*rad_wheel * dt, 0],
                      [0, dt],
                      [0,  0]])
        
        
        F = getJacobofF(self,dt)
        
        # 4x1 mean state at time t
        predicted_x = f.dot(self._x) + b.dot(self._u)    
        
        # 4x4 state covariance matrix at time t
        predicted_P = F.dot(self._P).dot(F.T) + self._Q

        self._P = predicted_P
        self._x = predicted_x

    def update(self, meas_value, flag:int):
        """ 
        actual measurement
            z
        measurement prediction
            h(xt_pred)
        measurement residual
            y = z - h(xt_pred)
        innovation covariance
            S = H P Ht + R
        kalman gain
            K = P Ht S^-1
        mean correction
            x = x + K y
        covariance correction
            P = (I - K H) * P
        """
        #print(self._x)
        def getJacobofH(self, a=float, angle= float):
            theta = self._x[2]  
            H = np.array([[1,0,- a * sin(theta + angle), 0],
                          [0,1, a * cos(theta + angle), 0]], dtype= float)
            
            return H
        
        def get_h(self, a=float, angle= float):
            theta = self._x[2]          
            h = np.squeeze(np.array([[self._x[0] + a * cos(theta + angle)],
                                     [self._x[1] + a* sin(theta + angle)]]),1)
            return h
        
        
        if flag == 0:
            a, angle = 0.2, pi/2
        if flag == 1:
            a, angle = 1, pi
        if flag == 2:
            a, angle = 1, pi/4
        if flag == 3:
            a, angle = 1, -pi/4
            
        # 2x1 measurement prediction matrix
        h = get_h(self, a, angle)
        # 2x4 Jacobian Measurement Observation Matrix
        H = getJacobofH(self, a, angle)
        
        
        # 2x1 measurement matrix at time t ([[mDx1],[mDy1]])
        self._z = meas_value 

        # 2x1 innovation i.e (difference between the actual measurement z and the predicted measurement Hx)
        y = self._z - h
        
        # 2x2 innovation covariance matrix (a measure of the error between the predicted measurement and the actual measurement)
        S = H.dot(self._P).dot(H.T) + self._R
        
        # 4x2 Kalman Gain Matrix 
        K = self._P.dot(H.T).dot(np.linalg.inv(S.astype('float64')))

        # 2x1 updated state matrix after measurement at time t
        updated_x = self._x + K.dot(y)
        
        # 4x4 updated state covariance matrix after measurement at time t
        updated_P = (np.eye(num_states) - K.dot(H)).dot(self._P)

        self._x = updated_x
        self._P = updated_P   
        print(self._x)

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self) -> float:
       return np.sqrt(np.square(self._x[0])+np.square(self._x[1]))

    @property
    def omega(self) -> float:
        return self._u[0]
    