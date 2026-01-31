# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:33:55 2023

@author: Z0168020
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math


num_states = 3
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
        # initialized as 3x1 matrix [pos X, pos Y, theta]
        self._x = initial_state.reshape(num_states,1)
        # covariance of state Gaussian Random Variable (GRV)
        # initialized as 3x3 identity matrix
        self._P = initial_covariance  
        # control input matrix
        # initialized as 2x1 matrix [omega, theta dot]
        self._u = control_input
        # process noise matrix
        # initialized as 3x3 matrix diag([sigma(posX)2, sigma(posY)2, sigma(theta)2])
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
            theta = self._x[2]
            F = np.array([[1,0, self._u[0]*(-math.sin(theta))*rad_wheel*dt],
                          [0, 1, self._u[0]*(math.cos(theta))*rad_wheel*dt],
                          [0, 0, 1]], dtype = float)
            
            return F
        
        theta = self._x[2]
                      
        # 3x3 State transition matrix
        f = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        
        # 3x2 Control matrix 
        b = np.array([[math.cos(theta)*rad_wheel * dt, 0],
                      [math.sin(theta)*rad_wheel * dt, 0],
                      [0, dt]], dtype = float)
        
        # 3x3 Jacobian of state transition matrix
        F = getJacobofF(self,dt)
        
        # 3x1 mean state at time t
        predicted_x = f.dot(self._x) + b.dot(self._u)    
        
        # 3x3 state covariance matrix at time t
        predicted_P = F.dot(self._P).dot(F.T) + self._Q

        self._P = predicted_P
        self._x = predicted_x

    def update(self, meas_value, dt:float):
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
        

        def getJacobofH(self,dt):
            theta = self._x[2]
            H = np.array([[1,0,-math.sin(theta + math.pi)],
                          [0,1,math.cos(theta + math.pi)],
                          [1,0,-math.sin(theta + math.pi/4)],
                          [0,1,math.cos(theta + math.pi/4)],
                          [1,0,-math.sin(theta - math.pi/4)],
                          [0,1,math.cos(theta - math.pi/4)]])
            
            return H

        theta = self._x[2]
        
        # 6x1 measurement prediction matrix 
        h = np.squeeze(np.array([[self._x[0] - math.cos(theta + math.pi)],
                                 [self._x[1] - math.sin(theta + math.pi)],
                                 [self._x[0] - math.cos(theta + math.pi/4)],
                                 [self._x[1] - math.sin(theta + math.pi/4)],
                                 [self._x[0] - math.cos(theta - math.pi/4)],
                                 [self._x[1] - math.sin(theta - math.pi/4)]]),1)
        
        # 6x3 Jacobian Measurement Observation Matrix
        H = getJacobofH(self, dt)
        
        # 6x1 measurement matrix at time t ([[mDx1],[mDy1],[mDx2],[mDy2],[mDx3],[mDy3]])
        self._z = meas_value 

        # 6x1 measurement residual i.e (difference between the actual measurement z and the predicted measurement Hx)
        y = self._z - h
        
        # 6x6 innovation covariance matrix (a measure of the error between the predicted measurement and the actual measurement)
        S = H.dot(self._P).dot(H.T) + self._R
        
        # 3x6 Kalman Gain Matrix 
        K = self._P.dot(H.T).dot(np.linalg.inv(S.astype('float64')))

        # 4x1 updated state matrix after measurement at time t
        updated_x = self._x + K.dot(y)
        
        # 4x4 updated state covariance matrix after measurement at time t
        updated_P = (np.eye(3) - K.dot(H)).dot(self._P)

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
    
    
    
plt.ion()
plt.figure(figsize=(8,12))

init_state = np.array([11.5,49.5,1.15]).reshape(num_states,1)
init_cov = np.eye(num_states)
u = np.array([35.40, 
              0.005]).reshape(2,1)
process_variance = np.array([[0.5,  0,  0],
                             [0,  0.5,  0],
                             [0,    0,  0.005]])
measu_variance = 0.1
Q = np.eye(num_states).dot(process_variance)       
R = np.eye(measu_inputs).dot(measu_variance)
z = np.array([0,0,0,0,0,0]).reshape(measu_inputs,1)       


# initialize the class EKF with constructor parameters
ekf = EKF(init_state, init_cov, z, u, Q, R)

# minimum time step (dt) used in the simulation
time_step = 1
# total number of time steps for which simulation will run
total_time_steps = 100
 # time step at which measurement is received
measurement_rate = 3

true_mean_vehicle = np.zeros(( num_states, 1, total_time_steps))
true_control_input = np.zeros(( control_inputs, 1, total_time_steps))

true_mean_vehicle[:,:,0] = np.array([20,
                                     50,
                                      1.2]).reshape(num_states,1)
true_control_input[:,:,0] = np.array([36.364,
                                       0.005]).reshape(control_inputs,1)

mean_state_estimate_1 = np.zeros(( num_states, 1, total_time_steps))  #property of kalman_filter class

mean_after_prediction = np.zeros(( num_states, 1, total_time_steps))
mean_after_gps_update = np.zeros(( num_states, 1, total_time_steps))
updatesteps= []

for step in range(total_time_steps):

    if step>0:        
        
        theta = true_mean_vehicle[2,:,step-1] + true_control_input[1,:,step-1]*time_step
        true_mean_vehicle[:,:,step] = true_mean_vehicle[:,:,step-1] +  np.vstack((time_step * true_control_input[0,:,step-1] * np.array([[math.cos(theta)],[math.sin(theta)]]) * rad_wheel, true_control_input[1,:,step-1]*time_step))         
 
        if step > 50 and step < 60:
            true_control_input[0,:,step] = 0.97*(true_control_input[0,:,step-1])
        
        if step > 25 and step < 40:
            true_control_input[1,:,step] = (true_control_input[1,:,step-1])-0.001
            
        if step > 45 and step < 70:
                true_control_input[1,:,step] = (true_control_input[1,:,step-1])+0.001
        
        else:
            true_control_input[0,:,step] = true_control_input[0,:,step-1]
            true_control_input[1,:,step] = true_control_input[1,:,step-1]
            
        ekf.predict(dt=time_step)
        
    
        if step % measurement_rate == 0:
            updatesteps.append(step)
            # mean before update
            mean_after_prediction[:,:,step] = ekf.mean
            ''' 
            Update by adding the random Gaussian noise into the true mean position
            '''
    
            mgps1 = (true_mean_vehicle[0:2,:,step] + np.array([[math.cos(theta+math.pi)],[math.sin(theta+math.pi)]]) + np.random.randn() * np.sqrt(measu_variance))
            mgps2 = (true_mean_vehicle[0:2,:,step] + np.array([[math.cos(theta+math.pi/4)],[math.sin(theta+math.pi/4)]]) + np.random.randn() * np.sqrt(measu_variance))
            mgps3 = (true_mean_vehicle[0:2,:,step] + np.array([[math.cos(theta-math.pi/4)],[math.sin(theta-math.pi/4)]])  + np.random.randn() * np.sqrt(measu_variance))
            
    
            measurement_update = np.vstack((mgps1,mgps2,mgps3))
    
            ekf.update(meas_value=measurement_update, dt=time_step)
            
            # mean after update
            mean_after_gps_update[:,:,step] = ekf.mean
            
    mean_state_estimate_1[:,:,step] = ekf.mean
    
updatesteps = np.array(updatesteps)

plt.subplot(3, 1, 1)
plt.title('PositionXGPS')
plt.plot(true_mean_vehicle[0,:,updatesteps], 'ko--', label = "True")
plt.plot(mean_after_prediction[0,:,updatesteps], 'co--', label = "Pred")
plt.plot(mean_after_gps_update[0,:,updatesteps], 'ro--', label = "Corr")
plt.legend(loc='lower right')

plt.subplot(3, 1, 2)
plt.title('PositionYGPS')
plt.plot(true_mean_vehicle[1,:,updatesteps], 'ko--', label = "True")
plt.plot(mean_after_prediction[1,:,updatesteps], 'co--', label = "Pred")
plt.plot(mean_after_gps_update[1,:,updatesteps], 'ro--', label = "Corr")
plt.legend(loc='lower right')

plt.subplot(3, 1, 3)
plt.title('Heading')
plt.plot(true_mean_vehicle[2,:,updatesteps], 'ko--', label = "True")
plt.plot(mean_after_prediction[2,:,updatesteps], 'co--', label = "Pred")
plt.plot(mean_after_gps_update[2,:,updatesteps], 'ro--', label = "Corr")
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
