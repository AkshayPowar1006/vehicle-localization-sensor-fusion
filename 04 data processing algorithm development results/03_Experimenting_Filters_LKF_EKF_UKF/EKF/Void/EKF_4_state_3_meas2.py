# -*- coding: utf-8 -*-
"""
Created on Tue May  9 14:44:29 2023

@author: Z0168020
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math


# offsets of each variable in the mean state matrix
state_Dx, state_Dy, state_theta, state_theta_dot = 0,1,2,3
num_states = max(state_Dx, state_Dy, state_theta, state_theta_dot) + 1
rad_wheel =  0.055

class EKF:

    def __init__(self, initial_pos, 
                       initial_omega,
                       initial_theta,
                       initial_theta_dot,
                       variance) -> None:
 
        # mean state matrix of Gaussian Random Variable (GRV)
        # initialized as 3x1 matrix
        self._x = np.zeros(num_states).reshape(num_states,1) 
        self._x[state_Dx] = initial_pos[0]
        self._x[state_Dy] = initial_pos[1]
        self._x[state_theta] = initial_theta
        self._x[state_theta_dot] = initial_theta_dot
        #self._theta_dot = initial_theta_dot
        self._omega = initial_omega
        
        # noise input matrix
        # initialized as 2x2 matrix
        self._accel_variance = np.eye(4)*variance
        self._accel_variance[2,2] = variance/100
        self._accel_variance[3,3] = variance/10000

        # covariance of state Gaussian Random Variable (GRV)
        # initialized as # 4x4 identity matrix
        self._P = np.eye(num_states)  
        

    def predict(self, dt: float) -> None:
        """     
        Equations related to prediction Step
        x = Fx + Bu
        P = F P Ft + a
        """
        
        def getJacobofF(self,dt:float):    
            phi = self._x[2]  #+ self._x[3]*dt
            F = np.array([[1,0, self._omega*(-math.sin(phi))*rad_wheel*dt, self._omega*(-math.sin(phi))*rad_wheel*dt*dt],
                          [0, 1, self._omega*(math.cos(phi))*rad_wheel*dt, self._omega*( math.cos(phi))*rad_wheel*dt*dt],
                          [0, 0, 1, dt],
                          [0, 0, 0, 1]], dtype=float)
            
            return F
        
        phi = self._x[2] #+ self._x[3]*dt
        u = np.array([self._omega, self._x[3]], dtype=float).reshape(2,1)
        # 3x3 State transition matrix
        f = np.eye(4)
        
        # 3x2 Control Jacobian w.r.t. omega and theta_dot
        b = np.array([[math.cos(phi)*rad_wheel * dt, 0],
                      [math.sin(phi)*rad_wheel * dt, 0],
                      [0, dt],
                      [0,  0]],dtype= float)
        
        
        F = getJacobofF(self,dt)
        
        # 4x1 mean state at time t
        predicted_x = f.dot(self._x) + b.dot(u)    
        
        # 4x4 state covariance matrix at time t
        predicted_P = F.dot(self._P).dot(F.T) + self._accel_variance

        self._P = predicted_P
        self._x = predicted_x

    def update(self, meas_value, meas_variance,dt:float):
        """       
        Equations related to update Step
        y = z - h(xt_pred)
        S = H P Ht + R
        K = P Ht S^-1
        x = x + K y
        P = (I - K H) * P
        """
        
        def getJacobofH(self,dt):
            phi = self._x[2]  
            H = np.array([[1,0,-math.sin(phi + math.pi), 0],
                          [0,1,math.cos(phi + math.pi), 0],
                          [1,0,-math.sin(phi + math.pi/4), 0],
                          [0,1,math.cos(phi + math.pi/4), 0],
                          [1,0,-math.sin(phi - math.pi/4), 0],
                          [0,1,math.cos(phi - math.pi/4), 0]], dtype = float)
            
            return H

        phi = self._x[2]  
        
        h = np.squeeze(np.array([[self._x[0] - math.cos(phi + math.pi)],
                                 [self._x[1] - math.sin(phi + math.pi)],
                                 [self._x[0] - math.cos(phi + math.pi/4)],
                                 [self._x[1] - math.sin(phi + math.pi/4)],
                                 [self._x[0] - math.cos(phi - math.pi/4)],
                                 [self._x[1] - math.sin(phi - math.pi/4)]]),1)
        
        # 6x3 Jacobian Measurement Observation Matrix
        H = getJacobofH(self, dt)
        
        # 6x1 measurement matrix at time t ([[mDx1],[mDy1],[mDx2],[mDy2],[mDx3],[mDy3]])
        z = meas_value 
        
        # 6x6 measurement noise covariance matrix ([[sigma_x^2, 0], [0, sigma_y^2]])
        R = meas_variance

        # 6x1 innovation i.e (difference between the actual measurement z and the predicted measurement Hx)
        y = z - h
        
        # 6x6 innovation covariance matrix (a measure of the error between the predicted measurement and the actual measurement)
        S = H.dot(self._P).dot(H.T) + R
        
        # 3x1 Kalman Gain Matrix 
        K = self._P.dot(H.T).dot(np.linalg.inv(S.astype('float64')))

        # 4x1 updated state matrix after measurement at time t
        updated_x = self._x + K.dot(y)
        
        # 4x4 updated state covariance matrix after measurement at time t
        updated_P = (np.eye(num_states) - K.dot(H)).dot(self._P)

        self._x = updated_x
        self._P = updated_P   
        #print(self._x)

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
    def omega(self) -> float:
        return self._omega
    
    
    
plt.ion()
plt.figure(figsize=(12,18))

# minimum time step (dt) used in the simulation
time_step = 1
# total number of time steps for which simulation will run
total_time_steps = 120
 # time step at which measurement is received
measurement_rate = 3

true_mean_vehicle = np.zeros(( num_states, 1, total_time_steps))
true_omega = np.full((total_time_steps,), 0.0)
#true_theta_dot = np.full((total_time_steps,), 0.0)

# defining the initial true mean state of a system at time t=0
true_theta = 1.2  #68.76 degrees
true_theta_dot = 0.005 # 2  degrees
true_omega[0] = 36.36

true_mean_vehicle[:,:,0] = np.array([[20],[50],[true_theta],[true_theta_dot]])

# sensor measurement variance matrix
sigma_measurement = 0.000001
measurement_variance = np.eye(6)* sigma_measurement

# initialize the class KF with constructor parameters
ekf = EKF(initial_pos=np.array([[20],[50]]),
                      initial_omega=36.364,
                      initial_theta = np.array([1.2]),
                      initial_theta_dot = np.array([0.005]),
                      variance= 0.005)

mean_state_estimate_1 = np.zeros(( num_states, 1, total_time_steps))  #property of kalman_filter class

mean_after_prediction = []
mean_after_gps_update = []
updatesteps= []

for step in range(total_time_steps):

    if step>0:        
        
        phi = true_mean_vehicle[2,:,step-1] 
        
        true_mean_vehicle[:,:,step] = true_mean_vehicle[:,:,step-1] +  np.vstack((time_step * true_omega[step-1] * np.array([[math.cos(phi)],[math.sin(phi)]]) * rad_wheel,
                                                                                  true_mean_vehicle[3,0,step-1]*time_step,
                                                                                  0))  
        
        if step > 50 and step < 70:
            true_omega[step] = 0.97*(true_omega[step-1])
        
        if step > 50 and step < 70:
            #true_omega[step] =1.03*(true_omega[step-1])
            true_mean_vehicle[3,0,step] = (true_mean_vehicle[3,0,step-1])-0.001
        else:
            true_omega[step] = true_omega[step-1]
            true_mean_vehicle[3,0,step] = true_mean_vehicle[3,0,step-1]

            
        ekf.predict(dt=time_step)
        
    
        if step % measurement_rate == 0:
            updatesteps.append(step)
            print(step)
            # mean before update
            mean_after_prediction.append(ekf.mean)
            ''' 
            Update by adding the random Gaussian noise into the true mean position
            '''
            mgps1 = (true_mean_vehicle[0:2,:,step] + np.array([[math.cos(phi+math.pi)],[math.sin(phi+math.pi)]]) + np.random.randn() * np.sqrt(sigma_measurement))
            mgps2 = (true_mean_vehicle[0:2,:,step] + np.array([[math.cos(phi+math.pi/4)],[math.sin(phi+math.pi/4)]]) + np.random.randn() * np.sqrt(sigma_measurement))
            mgps3 = (true_mean_vehicle[0:2,:,step] + np.array([[math.cos(phi-math.pi/4)],[math.sin(phi-math.pi/4)]])  + np.random.randn() * np.sqrt(sigma_measurement))
            
    
            measurement_update = np.vstack((mgps1,mgps2,mgps3))
    
            ekf.update(meas_value=measurement_update, meas_variance=measurement_variance, dt=time_step)
            
            # mean after update
            mean_after_gps_update.append(ekf.mean)

    mean_state_estimate_1[:,:,step] = ekf.mean
    
mean_after_prediction = np.array(mean_after_prediction)
mean_after_gps_update = np.array(mean_after_gps_update)
updatesteps = np.array(updatesteps)

plt.subplot(4, 1, 1)
plt.title('PositionXGPS')
plt.plot(true_mean_vehicle[0,:,updatesteps[:]], 'ko--', label = "True")
plt.plot(mean_after_prediction[:,0,0], 'co--', label = "Pred")
plt.plot(mean_after_gps_update[:,0,0], 'ro--', label = "Corr")
plt.legend(loc='lower right')

plt.subplot(4, 1, 2)
plt.title('PositionYGPS')
plt.plot(true_mean_vehicle[1,:,updatesteps[:]], 'ko--', label = "True")
plt.plot(mean_after_prediction[:,1,0], 'co--', label = "Pred")
plt.plot(mean_after_gps_update[:,1,0], 'ro--', label = "Corr")
plt.legend(loc='lower right')

n = 0
plt.subplot(4, 1, 3)
plt.title('Heading')
plt.plot(true_mean_vehicle[2,:,updatesteps[n:]], 'ko--', label = "True")
plt.plot(mean_after_prediction[n:,2,0], 'co--', label = "Pred")
plt.plot(mean_after_gps_update[n:,2,0], 'ro--', label = "Corr")
plt.legend(loc='upper right')

plt.subplot(4, 1, 4)
plt.title('Heading Change')
plt.plot(true_mean_vehicle[3,:,updatesteps[n:]], 'ko--', label = "True")
plt.plot(mean_after_prediction[n:,3,0], 'co--', label = "Pred")
plt.plot(mean_after_gps_update[n:,3,0], 'ro--', label = "Corr")
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()
