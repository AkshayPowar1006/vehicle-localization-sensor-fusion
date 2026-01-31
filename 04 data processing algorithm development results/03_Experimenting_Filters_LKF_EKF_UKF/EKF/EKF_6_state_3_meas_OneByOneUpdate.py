# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:44:29 2023

@author: Z0168020
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi
from functions_joseph import next_state_JosephModel 
import random

num_states = 6
measu_inputs = 2
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
        
               
        x_t_0 = self._x[0]
        y_t_0 = self._x[1]
        v_t_0 = self._x[2]
        theta_t_0 = self._x[3]
        a_t_0 = self._x[4]
        theta_dot_t_0 = self._x[5]
        
        v_t_1 = v_t_0 + a_t_0 * dt
        theta_t_1 = theta_t_0 + theta_dot_t_0 * dt
        a_t_1 = a_t_0
        theta_dot_t_1 = theta_dot_t_0
        
        if theta_dot_t_0 == 0:
            x_t_1 =  x_t_0 + cos(theta_t_0)*(v_t_0*dt + a_t_0*dt*dt/2)
            y_t_1 =  y_t_0 + sin(theta_t_0)*(v_t_0*dt + a_t_0*dt*dt/2)
        else:
            x_t_1 = x_t_0 + (v_t_1 * (sin(theta_t_1) - sin(theta_t_0)) + a_t_0 * sin(theta_t_0) * dt)/theta_dot_t_0 - a_t_0 * (cos(theta_t_0) - cos(theta_t_1)) / (theta_t_0)**2
            y_t_1 = y_t_0 + (v_t_1 * (cos(theta_t_0) - cos(theta_t_1)) - a_t_0 * cos(theta_t_0) * dt)/theta_dot_t_0 - a_t_0 * (sin(theta_t_0) - sin(theta_t_1)) / (theta_t_0)**2
        
        
        ################# X ######################
        dx_dx = 1
        dx_dy = 0
        if theta_dot_t_0 != 0:
            dx_dv = (sin(theta_t_1) - sin(theta_t_0))/ theta_dot_t_0
            dx_dtheta = (v_t_0 * (cos(theta_t_1) - cos(theta_t_0)) + a_t_0 * cos(theta_t_1) * dt) / theta_dot_t_0  + a_t_0 * (sin(theta_t_0) - sin(theta_t_1)) / (theta_t_0)**2
            dx_da = dt*sin(theta_t_1)/theta_dot_t_0 + (cos(theta_t_1) - cos(theta_t_0))/theta_dot_t_0**2
            dx_dtheta_dot = (dt*(v_t_0 + a_t_0 * dt) * cos(theta_t_1)) / theta_dot_t_0 + (v_t_0*(sin(theta_t_0) - sin(theta_t_1)) - 2*a_t_0*dt*sin(theta_t_1)) / theta_dot_t_0**2  + (2*a_t_0 *(cos(theta_t_0) - cos(theta_t_1))) / theta_dot_t_0**3
        else:
            dx_dv = cos(theta_t_0) * dt
            dx_dtheta = - sin(theta_t_0)*(v_t_0*dt + a_t_0*dt*dt/2)
            dx_da = cos(theta_t_0)*dt*dt/2
            dx_dtheta_dot = 0
               
        ################# Y ######################
        dy_dx = 0
        dy_dy = 1
        if theta_dot_t_0 != 0:
            dy_dv = (cos(theta_t_0) - cos(theta_t_1))/ theta_dot_t_0
            dy_dtheta = (v_t_0 * (sin(theta_t_1) - sin(theta_t_0)) + a_t_0 * sin(theta_t_1) * dt) / theta_dot_t_0  + a_t_0 * (cos(theta_t_1) - cos(theta_t_0)) / (theta_t_0)**2
            dy_da = -dt*cos(theta_t_1)/theta_dot_t_0 + (sin(theta_t_1) - sin(theta_t_0))/theta_dot_t_0**2
            dy_dtheta_dot = (dt*(v_t_0 + a_t_0 * dt) * sin(theta_t_1)) / theta_dot_t_0 + (v_t_0 * (-cos(theta_t_0) + cos(theta_t_1)) + 2*a_t_0*dt*cos(theta_t_1)) / theta_dot_t_0**2  + (2*a_t_0 *(sin(theta_t_0) - sin(theta_t_1))) / theta_dot_t_0**3
        else:
            dy_dv = sin(theta_t_0) * dt
            dy_dtheta = cos(theta_t_0)*(v_t_0*dt + a_t_0*dt*dt/2)
            dy_da = sin(theta_t_0)*dt*dt/2
            dy_dtheta_dot = 0
            
        ################# V ######################
        
        dv_dx = 0
        dv_dy = 0
        dv_dv = 1
        dv_dtheta = 0
        dv_da = dt
        dv_dtheta_dot = 0
        
        ################# THETA ######################
        dtheta_dx = 0
        dtheta_dy = 0
        dtheta_dv = 0
        dtheta_dtheta = 1
        dtheta_da = 0
        dtheta_dtheta_dot = dt
        
        ################# A ######################
        da_dx = 0
        da_dy = 0
        da_dv = 0
        da_dtheta = 0
        da_da = 1
        da_dtheta_dot = 0
        
        ################# THETA_DOT ###############
        dtheta_dot_dx = 0
        dtheta_dot_dy = 0
        dtheta_dot_dv = 0
        dtheta_dot_dtheta = 0
        dtheta_dot_da = 0
        dtheta_dot_dtheta_dot = 1
        
        
        
        F = np.array([[        dx_dx,         dx_dy,         dx_dv,         dx_dtheta,         dx_da,         dx_dtheta_dot],
                      [        dy_dx,         dy_dy,         dy_dv,         dy_dtheta,         dy_da,         dy_dtheta_dot],
                      [        dv_dx,         dv_dy,         dv_dv,         dv_dtheta,         dv_da,         dv_dtheta_dot],
                      [    dtheta_dx,     dtheta_dy,     dtheta_dv,     dtheta_dtheta,     dtheta_da,     dtheta_dtheta_dot],
                      [        da_dx,         da_dy,         da_dv,         da_dtheta,         da_da,         da_dtheta_dot],
                      [dtheta_dot_dx, dtheta_dot_dy, dtheta_dot_dv, dtheta_dot_dtheta, dtheta_dot_da, dtheta_dot_dtheta_dot]])
        
        
        # 4x1 mean state at time t
        predicted_x = np.array([x_t_1, y_t_1, v_t_1, theta_t_1, a_t_1, theta_dot_t_1]).reshape(6,1)    
        predicted_x[3] = self.wraptopi(predicted_x[3])
        
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
            H = np.array([[1,0,0,-1 * a * sin(theta + angle),0, 0],
                          [0,1,0, a * cos(theta + angle),0, 0]], dtype= float)   
            return H

        #theta = self._x[2]  
        
        def get_h(self, a=float, angle= float):
            theta = self._x[2]          
            h = np.squeeze(np.array([[self._x[0] + a * cos(theta + angle)],
                                     [self._x[1] + a* sin(theta + angle)]]),1)
            return h
        
        if flag == 0:
            a, angle = 0.2, 0
        if flag == 1:
            a, angle = 1, pi
        if flag == 2:
            a, angle = 1, pi/4
        if flag == 3:
            a, angle = 1, -pi/4
        
        # 2x1 measurement prediction matrix
        h = get_h(self, a, angle)
        
        # 6x3 Jacobian Measurement Observation Matrix
        H = getJacobofH(self, a, angle)
        
        # 6x1 measurement matrix at time t ([[mDx1],[mDy1],[mDx2],[mDy2],[mDx3],[mDy3]])
        self._z = meas_value 

        # 6x1 innovation i.e (difference between the actual measurement z and the predicted measurement Hx)
        y = self._z - h
        
        # 6x6 innovation covariance matrix (a measure of the error between the predicted measurement and the actual measurement)
        S = H.dot(self._P).dot(H.T) + self._R
        
        # 4x6 Kalman Gain Matrix 
        K = self._P.dot(H.T).dot(np.linalg.inv(S.astype('float64')))

        # 4x1 updated state matrix after measurement at time t
        updated_x = self._x + K.dot(y)
        updated_x[3] = self.wraptopi(updated_x[3]) 
        
        # 4x4 updated state covariance matrix after measurement at time t
        updated_P = (np.eye(num_states) - K.dot(H)).dot(self._P)

        self._x = updated_x
        self._P = updated_P   
        print(self._x)

    

    def wraptopi(self, angle):
        while angle > pi:
            angle -= 2 * pi
        while angle < -1*pi:
            angle += 2 * pi
        return angle

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


if __name__ == '__main__':
        
    
    
    init_state = np.array([20, 50, 1.0, 1.2, 0.01, 0.005]).reshape(num_states,1)
    init_cov = np.eye(num_states)
    u = np.array([0.01, 
                  0.005]).reshape(2,1)
    process_variance = np.array([[0.15,  0,    0,    0,     0,      0],
                                 [0,  0.15,    0,    0,     0,      0],
                                 [0,    0, 0.015,    0,     0,      0],
                                 [0,    0,    0, 0.015,     0,      0],
                                 [0,    0,    0,    0, 0.0015,      0],
                                 [0,    0,    0,    0,     0, 0.00015]])
    measu_variance = 0.1
    Q = np.eye(num_states).dot(process_variance)       
    R = np.eye(measu_inputs).dot(measu_variance)
    z = np.array([0,0]).reshape(measu_inputs,1)    
        
    
    # initialize the class EKF with constructor parameters
    ekf = EKF(init_state, init_cov, z, u, Q, R)  
    
    # minimum time step (dt) used in the simulation
    time_step = 1
    # total number of time steps for which simulation will run
    total_time_steps = 120
     # time step at which measurement is received
    measurement_rate = 3
    
    true_mean_vehicle = np.zeros(( num_states, 1, total_time_steps))
    true_control_input = np.zeros(( control_inputs, 1, total_time_steps))
    
    true_mean_vehicle[:,:,0] = np.array([20,
                                         50,
                                         1.0,
                                         1.2,
                                         0.01,
                                         0.005]).reshape(num_states,1)
    true_control_input[:,:,0] = np.array([0.01,
                                           0.005]).reshape(control_inputs,1)
    
    mean_state_estimate_1 = np.zeros(( num_states, 1, total_time_steps))  #property of kalman_filter class
    
    mean_after_prediction = np.zeros(( num_states, 1, total_time_steps))
    mean_after_gps_update = np.zeros(( num_states, 1, total_time_steps))
    updatesteps= []
    
    for step in range(total_time_steps):
    
        if step>0:        
             
                                                                                           
            if step > 00 and step < 20:
                #true_control_input[0,:,step] = random.uniform(-1 * true_control_input[0,:,0], true_control_input[0,:,0])
                true_control_input[0,:,step] = 0.97*(true_control_input[0,:,step-1])
            else:
                #true_control_input[0,:,step] = true_control_input[0,:,step-1]
                true_control_input[0,:,step] = true_control_input[0,:,step-1]
            if step > 00 and step < 20:
                #true_control_input[1,:,step] = random.uniform(-1 * true_control_input[1,:,0], true_control_input[1,:,0])
                true_control_input[1,:,step] = (true_control_input[1,:,step-1])-0.001
            else:
                #true_control_input[1,:,step] = true_control_input[1,:,step-1]
                true_control_input[1,:,step] = true_control_input[1,:,step-1]
            
            theta = true_mean_vehicle[3,:,step-1] + true_control_input[1,:,step-1]*time_step
            
            
            true_mean_vehicle[:,:,step] = next_state_JosephModel(true_mean_vehicle[:,:,step-1], time_step)   
            true_mean_vehicle[4,:,step] = true_control_input[0,:,step]                                  
            true_mean_vehicle[5,:,step] = true_control_input[1,:,step]
            ekf._x[4] = true_control_input[0,:,step]   
            ekf._x[5] = true_control_input[1,:,step]
    
            ekf.predict(dt=time_step)
            
        
            if step % measurement_rate == 0:
                updatesteps.append(step)
                # mean before update
                mean_after_prediction[:,:,step] = ekf.mean
                ''' 
                Update by adding the random Gaussian noise into the true mean position
                '''
        
                mgps1 = (true_mean_vehicle[0:2,:,step] + np.array([[cos(theta+pi)],[sin(theta+pi)]]) + np.random.randn() * np.sqrt(measu_variance))
                mgps2 = (true_mean_vehicle[0:2,:,step] + np.array([[cos(theta+pi/4)],[sin(theta+pi/4)]]) + np.random.randn() * np.sqrt(measu_variance))
                mgps3 = (true_mean_vehicle[0:2,:,step] + np.array([[cos(theta-pi/4)],[sin(theta-pi/4)]])  + np.random.randn() * np.sqrt(measu_variance))
                
        
                #measurement_update = np.vstack((mgps1,mgps2,mgps3))
        
                ekf.update(meas_value=mgps1, flag=1)
                ekf.update(meas_value=mgps2, flag=2)
                ekf.update(meas_value=mgps3, flag=3)
                
                # mean after update
                mean_after_gps_update[:,:,step] = ekf.mean
    
        mean_state_estimate_1[:,:,step] = ekf.mean
        
    updatesteps = np.array(updatesteps)
    
    plt.ion()
    fig, ((ax1, ax2, ax3, ax4, ax5)) = plt.subplots(5, 1, figsize=(20, 15))
    
   
    ax1.set_title('Position X-Y GPS')
    ax1.plot(true_mean_vehicle[0,:,updatesteps],true_mean_vehicle[1,:,updatesteps], 'ko--', label = "True")
    ax1.plot(mean_after_prediction[0,:,updatesteps], mean_after_prediction[1,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    ax1.plot(mean_after_gps_update[0,:,updatesteps], mean_after_gps_update[1,:,updatesteps], color='salmon',marker='o', linestyle='--', label = "Corr")
    ax1.legend(loc='lower right')
    ax1.ticklabel_format(style='plain', useOffset=False)

    ax2.set_title('Velocity')
    ax2.plot(true_mean_vehicle[2,:,updatesteps], 'ko--', label = "True")
    ax2.plot(mean_after_prediction[2,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    ax2.plot(mean_after_gps_update[2,:,updatesteps],  color='salmon',marker='o', linestyle='--', label = "Corr")
    ax2.legend(loc='upper right')
    
    ax3.set_title('Orientation')
    ax3.plot(true_mean_vehicle[3,:,updatesteps], 'ko--', label = "True")
    ax3.plot(mean_after_prediction[3,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    ax3.plot(mean_after_gps_update[3,:,updatesteps], color='salmon',marker='o', linestyle='--', label = "Corr")
    ax3.legend(loc='upper right')
    
    ax4.set_title('Acceleration')
    ax4.plot(true_mean_vehicle[4,:,updatesteps], 'ko--', label = "True")
    ax4.plot(mean_after_prediction[4,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    ax4.plot(mean_after_gps_update[4,:,updatesteps], color='salmon',marker='o', linestyle='--', label = "Corr")
    ax4.legend(loc='upper right')
    
    ax5.set_title('Orientation Change')
    ax5.plot(true_mean_vehicle[5,:,updatesteps], 'ko--', label = "True")
    ax5.plot(mean_after_prediction[5,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    ax5.plot(mean_after_gps_update[5,:,updatesteps], color='salmon',marker='o', linestyle='--', label = "Corr")
    ax5.legend(loc='upper right')
    
    
    # plt.subplot(3, 2, 1)
    # plt.title('PositionXGPS')
    # plt.plot(true_mean_vehicle[0,:,updatesteps], 'ko--', label = "True")
    # plt.plot(mean_after_prediction[0,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    # plt.plot(mean_after_gps_update[0,:,updatesteps], color='salmon',marker='o', linestyle='--', label = "Corr")
    # plt.legend(loc='lower right')
    
    # plt.subplot(3, 2, 2)
    # plt.title('PositionYGPS')
    # plt.plot(true_mean_vehicle[1,:,updatesteps], 'ko--', label = "True")
    # plt.plot(mean_after_prediction[1,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    # plt.plot(mean_after_gps_update[1,:,updatesteps], color='salmon',marker='o', linestyle='--', label = "Corr")
    # plt.legend(loc='lower right')
    
    # plt.subplot(3, 2, 3)
    # plt.title('Velocity')
    # plt.plot(true_mean_vehicle[2,:,updatesteps], 'ko--', label = "True")
    # plt.plot(mean_after_prediction[2,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    # plt.plot(mean_after_gps_update[2,:,updatesteps],  color='salmon',marker='o', linestyle='--', label = "Corr")
    # plt.legend(loc='upper right')
    
    # plt.subplot(3, 2, 4)
    # plt.title('Orientation')
    # plt.plot(true_mean_vehicle[3,:,updatesteps], 'ko--', label = "True")
    # plt.plot(mean_after_prediction[3,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    # plt.plot(mean_after_gps_update[3,:,updatesteps], color='salmon',marker='o', linestyle='--', label = "Corr")
    # plt.legend(loc='upper right')
    
    
    # plt.subplot(3, 2, 5)
    # plt.title('Acceleration')
    # plt.plot(true_mean_vehicle[4,:,updatesteps], 'ko--', label = "True")
    # plt.plot(mean_after_prediction[4,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    # plt.plot(mean_after_gps_update[4,:,updatesteps], color='salmon',marker='o', linestyle='--', label = "Corr")
    # plt.legend(loc='upper right')
    
    # plt.subplot(3, 2, 6)
    # plt.title('Orientation Change')
    # plt.plot(true_mean_vehicle[5,:,updatesteps], 'ko--', label = "True")
    # plt.plot(mean_after_prediction[5,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    # plt.plot(mean_after_gps_update[5,:,updatesteps], color='salmon',marker='o', linestyle='--', label = "Corr")
    # plt.legend(loc='upper right')
    
    
    plt.tight_layout()
    plt.show()
