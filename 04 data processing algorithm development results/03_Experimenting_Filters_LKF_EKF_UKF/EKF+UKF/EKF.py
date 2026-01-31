# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:44:29 2023

@author: Z0168020
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, pi, exp, atan2


num_states = 4
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
        self._R0 = measurement_noise
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
        # def getJacobofF(self,dt:float):    
        #     theta = self._x[2]  #+ self._x[3]*dt
        #     F = np.array([[1,0, self._u[0]*(-sin(theta))*rad_wheel*dt, self._u[0]*(-sin(theta))*rad_wheel*dt*dt],
        #                   [0, 1, self._u[0]*(cos(theta))*rad_wheel*dt, self._u[0]*( cos(theta))*rad_wheel*dt*dt],
        #                   [0, 0, 1, dt],
        #                   [0, 0, 0, 1]],dtype=float)
            
        #     return F
        def getJacobofF(self,dt:float):    
            theta = self._x[2]  #+ self._x[3]*dt
            F = np.array([[1,0, self._u[0][0]*(-sin(theta))*rad_wheel*dt, 0],
                          [0, 1, self._u[0][0]*(cos(theta))*rad_wheel*dt, 0],
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
        predicted_x[2] = self.wraptopi(predicted_x[2])
        
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
            a, angle = 0.2, 0
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
        y = self._z - h #Here we are calculating Residual
        
        # 2x2 innovation covariance matrix (a measure of the error between the predicted measurement and the actual measurement)
        S = H.dot(self._P).dot(H.T) + self._R
        
        # if y[0]>3*self._R[0,0] or y[1]>3*self._R[1,1]:
        # #Here we have to calculate the MLDB distance
        #     MLB_dist = np.sqrt((y.T).dot(np.linalg.inv(S.astype('float64'))).dot(y))
        # #MLB_dist = np.sqrt((y.dot(y.T))).dot(np.linalg.inv(S.astype('float64')))
        #     Eps = 0.01 # Constant
        #     MLBD_weighted = 1/(1 + np.exp(-MLB_dist + Eps))
        
        #     delta = 20 #Constant 
        #     self._R = delta*MLBD_weighted * self._R
        # else:
        #       self._R = self._R0
        
        # 4x2 Kalman Gain Matrix 
        K = self._P.dot(H.T).dot(np.linalg.inv(S.astype('float64')))

        # 2x1 updated state matrix after measurement at time t
        updated_x = self._x + K.dot(y)
        updated_x[2] = self.wraptopi(updated_x[2]) 
        
        # 4x4 updated state covariance matrix after measurement at time t
        updated_P = (np.eye(num_states) - K.dot(H)).dot(self._P)

        self._x = updated_x
        self._P = updated_P   
        #print(self._x)
        
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
        

    
    init_state = np.array([20,50,1.2,0.005]).reshape(num_states,1)
    init_cov = np.eye(num_states)
    u = np.array([36.364, 
                  0.005]).reshape(2,1)
    process_variance = np.array([[1.5,  0,  0,     0],
                                  [0,  1.5,  0,     0],
                                  [0,    0,  0.0015, 0],
                                  [0,    0,    0,     0.000015]])
    measu_variance = 0.01
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
    measurement_rate = 5
    
    true_mean_vehicle = np.zeros(( num_states, 1, total_time_steps))
    true_control_input = np.zeros(( control_inputs, 1, total_time_steps))
    
    true_mean_vehicle[:,:,0] = np.array([20,
                                          50,
                                          1.2,
                                          0.005]).reshape(num_states,1)
    
    true_control_input[:,:,0] = np.array([36.364,
                                            0.005]).reshape(control_inputs,1)
    
    mean_state_estimate_1 = np.zeros(( num_states, 1, total_time_steps))  #property of kalman_filter class
    
    mean_after_prediction = np.zeros(( num_states, 1, total_time_steps))
    mean_after_gps_update = np.zeros(( num_states, 1, total_time_steps))
    updatesteps= []
    
    for step in range(total_time_steps):
    
        if step>0:        
             
                                                                                           
     
            if step > 00 and step < 20:
                true_control_input[0,:,step] = 0.97*(true_control_input[0,:,step-1])
            else:
                true_control_input[0,:,step] = true_control_input[0,:,step-1]
            
            if step > 00 and step < 20:
                true_control_input[1,:,step] = (true_control_input[1,:,step-1])-0.001
            else:
                true_control_input[1,:,step] = true_control_input[1,:,step-1]
            
            theta = true_mean_vehicle[2,:,step-1] + true_control_input[1,:,step-1]*time_step
            true_mean_vehicle[:,:,step] = true_mean_vehicle[:,:,step-1] +  np.vstack((time_step * true_control_input[0,:,step-1] * np.array([[cos(theta)],[sin(theta)]]) * rad_wheel,
                                                                                          true_control_input[1,:,step-1]*time_step,
                                                                                            0))                                                  
            true_mean_vehicle[3,:,step] = true_control_input[1,:,step-1]
            ekf.predict(dt=time_step)
            
        
            if step % measurement_rate == 0:
                updatesteps.append(step)
                print(step)
                # mean before update
                mean_after_prediction[:,:,step] = ekf.mean
                ''' 
                Update by adding the random Gaussian noise into the true mean position
                '''
        
                mgps1 = (true_mean_vehicle[0:2,:,step] + np.array([[cos(theta+pi)],[sin(theta+pi)]]) + np.random.randn() * np.sqrt(measu_variance))
                mgps2 = (true_mean_vehicle[0:2,:,step] + np.array([[cos(theta+pi/4)],[sin(theta+pi/4)]]) + np.random.randn() * np.sqrt(measu_variance))
                mgps3 = (true_mean_vehicle[0:2,:,step] + np.array([[cos(theta-pi/4)],[sin(theta-pi/4)]])  + np.random.randn() * np.sqrt(measu_variance))
                
        
                #measurement_update = np.vstack((mgps1,mgps2,mgps3))
                
                ekf.update(meas_value=mgps1, flag = 1)
                ekf.update(meas_value=mgps2, flag = 2)
                ekf.update(meas_value=mgps3, flag = 3)
                
                # mean after update
                mean_after_gps_update[:,:,step] = ekf.mean
    
        mean_state_estimate_1[:,:,step] = ekf.mean
        
    updatesteps = np.array(updatesteps)
    
    plt.ion()
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(10.5, 3.5))
  
    ax1.set_title('Position X-Y GPS')
    ax1.plot(true_mean_vehicle[0,:,updatesteps], true_mean_vehicle[1,:,updatesteps], 'ko--', label = "True")
    #ax1.plot(mean_after_prediction[0,:,updatesteps], mean_after_prediction[1,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    ax1.plot(mean_after_gps_update[0,:,updatesteps], mean_after_gps_update[1,:,updatesteps],  color='salmon',marker='o', linestyle='--', label = "Corr")
    ax1.legend(loc='lower right')
    ax1.set_xlabel('X-Position in m')
    ax1.set_ylabel('Y-Position in m')
    
    ax2.set_title('Heading')
    ax2.plot(true_mean_vehicle[2,:,updatesteps], 'ko--', label = "True")
    #ax2.plot(mean_after_prediction[2,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    ax2.plot(mean_after_gps_update[2,:,updatesteps],  color='salmon',marker='o', linestyle='--', label = "Corr")
    ax2.legend(loc='upper right')
    ax2.set_ylabel('Heading in rad')
    ax2.set_xlabel('Time in s')

    ax3.set_title('Heading Change')
    ax3.plot(true_mean_vehicle[3,:,updatesteps], 'ko--', label = "True")
    #ax3.plot(mean_after_prediction[3,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    ax3.plot(mean_after_gps_update[3,:,updatesteps],  color='salmon',marker='o', linestyle='--', label = "Corr")
    ax3.legend(loc='upper right')
    ax3.set_ylabel('Heading Change in rad/s')
    ax3.set_xlabel('Time in s')
    
    plt.tight_layout()
    plt.show()
    
    
    # plt.subplot(4, 1, 2)
    # plt.title('PositionYGPS')
    # plt.plot(true_mean_vehicle[1,:,updatesteps], 'ko--', label = "True")
    # plt.plot(mean_after_prediction[1,:,updatesteps], color='indigo',marker='o', linestyle='--', label = "Pred")
    # plt.plot(mean_after_gps_update[1,:,updatesteps],  color='salmon',marker='o', linestyle='--', label = "Corr")
    # plt.legend(loc='lower right')
