# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:10:40 2023

@author: Z0168020
"""
import numpy as np

# offsets of each variable in the mean state matrix
state_Dx, state_Dy, state_Vx, state_Vy = 0,1,2,3
num_states = max(state_Dx, state_Dy, state_Vx, state_Vy) + 1

class KF:
    """
    Implementation of the Linear Kalman filter.
    """
    def __init__(self, initial_x, 
                       initial_v,
                       accel_variance) -> None:
        
        """
        Initializes the mean state and covariance matrix of the state Gaussian random variable (GRV).
        
        Args:
            initial_x (list): The initial position of the system as a list of [x, y] coordinates.
            initial_v (list): The initial velocity of the system as a list of [vx, vy] coordinates.
            accel_variance (float): The variance of the acceleration noise.
        """
        
        # mean state matrix of Gaussian Random Variable (GRV)
        # initialized as 4x1 matrix
        self._x = np.zeros(num_states).reshape(num_states,1) 
        self._x[state_Dx] = initial_x[0]
        self._x[state_Dy] = initial_x[1]
        self._x[state_Vx] = initial_v[0]
        self._x[state_Vy] = initial_v[1]
        
        # noise input matrix
        # initialized as 2x2 matrix
        self._accel_variance = np.array([[accel_variance,0],[0,accel_variance]]) 

        # covariance of state Gaussian Random Variable (GRV)
        # initialized as # 4x4 identity matrix
        self._P = np.eye(num_states)  
        

    def predict(self, dt: float) -> None:
        """
        Predict the state of the object after a time interval of dt seconds.
        
        Parameters
        ----------
        dt : float
            The time interval in seconds.
            
        Returns
        -------
        None
        
        
        Equations related to prediction Step
        x = F x
        P = F P Ft + G a Gt
        """
        
        # 4x4 state transition matrix
        F = np.eye(num_states) 
        F[state_Dx,state_Vx] = dt     
        F[state_Dy,state_Vy] = dt
        '''
        F = array([[1., 0., 1., 0.],
                   [0., 1., 0., 1.],
                   [0., 0., 1., 0.],
                   [0., 0., 0., 1.]])
        '''
        # 4x1 mean state matrix at time t
        # x = F x --> new state as per the state transition matrix
        predicted_x = F.dot(self._x)
        
        # 4x2 control matrix
        G = np.zeros((4,2)) 
        G[state_Dx,state_Dx] =G[state_Dy,state_Dy] = dt**2/2
        G[state_Vx,state_Dx]= G[state_Vy,state_Dy] = dt

        # 4x4 state covariance matrix at time t
        predicted_P = F.dot(self._P).dot(F.T) + G.dot(self._accel_variance).dot(G.T)
        #print(predicted_P)

        self._P = predicted_P
        self._x = predicted_x

    def update(self, meas_value, meas_variance):
        """
        Update the state of the object after a time interval of dt seconds when measurements are received.
        
        Parameters
        ----------
        meas_value : array
            The actual measurement at a given time.
        meas_variance : array 
            The covariance matrix of the measurement noise.
        
        Returns
        -------
        None
        
        Equations related to update Step
        y = z - H x
        S = H P Ht + R
        K = P Ht S^-1
        x = x + K y
        P = (I - K H) * P
        """
        
        # 2x4 state observation matrix 
        H = np.concatenate((np.eye(2),np.zeros((2,2))),axis=1)

        # 2x1 measurement matrix at time t ([[mDx],[mDy]])
        z = meas_value 
        
        # 2x2 measurement noise covariance matrix ([[sigma_x^2, 0], [0, sigma_y^2]])
        R = meas_variance

        # 4x1 innovation i.e (difference between the actual measurement z and the predicted measurement Hx)
        y = z - H.dot(self._x)
        
        # 2x2 innovation covariance matrix (a measure of the error between the predicted measurement and the actual measurement)
        S = H.dot(self._P).dot(H.T) + R
        
        # 4x1 Kalman Gain Matrix 
        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        # 4x1 updated state matrix after measurement at time t
        updated_x = self._x + K.dot(y)
        
        # 4x4 updated state covariance matrix after measurement at time t
        updated_P = (np.eye(4) - K.dot(H)).dot(self._P)
        #print(updated_P)
        self._x = updated_x
        self._P = updated_P
        

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self) -> float:
        return self._x[state_Dx]

    @property
    def vel(self) -> float:
        return self._x[state_Vx]
