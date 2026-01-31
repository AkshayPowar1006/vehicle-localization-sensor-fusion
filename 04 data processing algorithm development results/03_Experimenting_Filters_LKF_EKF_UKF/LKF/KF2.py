# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:59:14 2023

@author: Z0168020
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# offsets of each variable in the mean state matrix
num_states = 6
measu_inputs = 2
control_inputs = 2
rad_wheel =  0.055
state_Dx, state_Dy, state_Vx, state_Vy, state_Ax, state_Ay = 0,1,2,3,4,5

class KF:
    """
    Implementation of the Linear Kalman filter.
    """
    def __init__(self, initial_state, 
                       initial_covariance,
                       measurement,
                       control_input,
                       process_noise,
                       measurement_noise) -> None:
        
        """
        Initializes the mean state and covariance matrix of the state Gaussian random variable (GRV).
        
        Args:
            initial_x (list): The initial position of the system as a list of [x, y] coordinates.
            initial_v (list): The initial velocity of the system as a list of [vx, vy] coordinates.
            accel_variance (float): The variance of the acceleration noise.
        """
        
        # mean state matrix of Gaussian Random Variable (GRV)
        # initialized as 4x1 matrix [posX, posY, velX, velY]
        self._x = initial_state.reshape(num_states,1)
        # covariance of state Gaussian Random Variable (GRV)
        # initialized as 4x4 identity matrix
        self._P = initial_covariance
        # control input matrix
        # initialized as 2x1 matrix [accX, accY]
        self._u = control_input
        # process noise matrix
        # initialized as 4x4 matrix diag([sigma(posX)2, sigma(posY)2, sigma(velX)2, sigma(velY)2])
        self._Q = process_noise
        # measurement noise matrix
        # initialized as 6x6 matrix diag([sigma(posX1)2, sigma(posY1)2)
        self._R = measurement_noise
        # measurement matrix
        # initialized as 2x1 empty matrix
        self._z = measurement  
        

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
        F[state_Dx,state_Ax] = dt**2/2
        F[state_Dy,state_Ay] = dt**2/2
        F[state_Vx,state_Ax] = dt     
        F[state_Vy,state_Ay] = dt
    
        B = np.zeros((num_states, control_inputs))
        B[state_Dx,state_Dx] = B[state_Dy,state_Dy] = dt**2/2
        B[state_Vx,state_Dx] = B[state_Vy,state_Dy] = dt
        B[state_Ax,state_Dx] = B[state_Ay,state_Dy] = 1
        # 4x1 mean state matrix at time t
        # x = F x --> new state as per the state transition matrix
        predicted_x = F.dot(self._x) 

        # 4x4 state covariance matrix at time t
        predicted_P = F.dot(self._P).dot(F.T) + B.dot(self._Q).dot(B.T)

        self._P = predicted_P
        self._x = predicted_x

    def update(self, meas_value):
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
        H = np.concatenate((np.eye(measu_inputs),np.zeros((measu_inputs,num_states - measu_inputs))),axis=1)

        # 2x1 measurement matrix at time t ([[mDx],[mDy]])
        self._z = meas_value 

        # 4x1 innovation i.e (difference between the actual measurement z and the predicted measurement Hx)
        y = self._z - H.dot(self._x)
        
        # 2x2 innovation covariance matrix (a measure of the error between the predicted measurement and the actual measurement)
        S = H.dot(self._P).dot(H.T) + self._R
        
        # 4x1 Kalman Gain Matrix 
        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        # 4x1 updated state matrix after measurement at time t
        updated_x = self._x + K.dot(y)
        
        # 4x4 updated state covariance matrix after measurement at time t
        updated_P = (np.eye(num_states) - K.dot(H)).dot(self._P)

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
        return (self._x[0]**2 + self._x[1]**2)**(1/2)

    @property
    def vel(self) -> float:
        return (self._x[2]**2 + self._x[3]**2)**(1/2)
    
    
    
def kalman_filter_localization(measurement_rate:int):
    
    plt.ion()
    #plt.figure(figsize=(15,9))
    #######################----------Initialization for filter--------------############################
    init_state = np.array([20, 50, 1, 1, 0.01, 0.01]).reshape(num_states,1)
    init_cov = np.eye(num_states)
    u = np.array([0, 0]).reshape(2,1)
    process_variance = 0.0005
    Q = np.eye(measu_inputs).dot(process_variance)     
    measu_variance = 0.33
    R = np.eye(measu_inputs).dot(measu_variance)
    z = np.array([0,0]).reshape(measu_inputs,1)    
    # initialize the class EKF with constructor parameters
    kf = KF(init_state, init_cov, z, u, Q, R)
    
    # minimum time step (dt), total number of time steps, measurement rate
    time_step, total_time_steps, measurement_rate = 1, 120, 1
    
    
    true_mean_vehicle = np.zeros(( num_states, 1, total_time_steps))
    true_control_input = np.zeros(( control_inputs, 1, total_time_steps))
    
    # defining the initial true mean state of a system at time t=0
    true_mean_vehicle[:,:,0] = np.array([20, 50, 2, 1,  0.01, 0.01]).reshape(num_states,1)
    true_control_input[:,:,0] = np.array([0.00, 0.00]).reshape(control_inputs,1)
    
    
    mean_state_estimate_1 = np.zeros((num_states, 1, total_time_steps))
   
    mean_state_after_gps1_update = np.zeros(( num_states, 1, total_time_steps))
    mean_state_after_gps1_update[:,:,0] = true_mean_vehicle[:,:,0]
    mean_state_after_gps2_update = np.zeros(( num_states, 1, total_time_steps))
    mean_state_after_gps2_update[:,:,0] = true_mean_vehicle[:,:,0]
    mean_state_after_gps3_update = np.zeros(( num_states, 1, total_time_steps))
    mean_state_after_gps3_update[:,:,0] = true_mean_vehicle[:,:,0]
    mean_state_after_prediction = np.zeros(( num_states, 1, total_time_steps))
    mean_state_after_prediction[:,:,0] = true_mean_vehicle[:,:,0]
    
    updatesteps= []
    
    F = np.eye(num_states) 
    F[state_Dx,state_Vx] = time_step     
    F[state_Dy,state_Vy] = time_step
    F[state_Dx,state_Ax] = time_step**2/2
    F[state_Dy,state_Ay] = time_step**2/2
    F[state_Vx,state_Ax] = time_step     
    F[state_Vy,state_Ay] = time_step

    for step in range(total_time_steps):

        if step>0:   
            
            true_mean_vehicle[:,:,step] = F.dot(true_mean_vehicle[:,:,step-1])
            
            kf.predict(dt=time_step)
    
            '''
            True mean position of the vehicle after minimum simulation time step. To calculate this 
            input velocity times the minimum time step is added to the previous true mean state.
            ---------------
            new position = old position + input velocity * simulation time step
            this is equivalent to 
            i = i + 2
            ---------------
            To simulate real-world conditions, random Gaussian noise is added to this position 
            to generate artificial position. This generated position is treated as sensor measurements.
            '''
            
            mean_state_after_prediction[:,:,step] = kf.mean
    
            if step != 0 and step % measurement_rate == 0:
                ''' Update by adding the random Gaussian noise into the true mean position'''
                updatesteps.append(step)
    
    
                kf.update(meas_value=((true_mean_vehicle[0:2,:,step] + np.array([[-1],[0]]) + 1 * np.random.randn() * np.sqrt(measu_variance))-np.array([[-1],[0]])))
                mean_state_after_gps1_update[:,:,step] = kf.mean
    
    
                kf.update(meas_value=((true_mean_vehicle[0:2,:,step] + np.array([[+0.5],[0.8660]]) + 1 * np.random.randn() * np.sqrt(measu_variance))-np.array([[+0.5],[0.8660]])))
                mean_state_after_gps2_update[:,:,step] = kf.mean
    
    
                kf.update(meas_value=(true_mean_vehicle[0:2,:,step] + np.array([[0.5],[-0.8660]]) + 1 * np.random.randn() * np.sqrt(measu_variance)-np.array([[0.5],[-0.8660]])))
                mean_state_after_gps3_update[:,:,step] = kf.mean
                
            
        mean_state_estimate_1[:,:,step] = kf.mean

    fig, ((ax1,ax3,ax5),(ax2,ax4,ax6),(ax7, ax8, ax9)) = plt.subplots(3,3, figsize=(10, 10))
    
    legend_labels = ['Estimated', 'True']
    legend_markers = ['salmon', 'm']
    
    ax1.set_title('Line Plot- Position X GPS')
    ax1.plot(mean_state_estimate_1[0,0,:],'salmon', label='Estimated')  
    ax1.plot(true_mean_vehicle[0,0,:], c='k', label = 'True')
    ax1.legend(loc = 'upper left')
    ax1.set_xlabel("Time in s")
    ax1.set_ylabel('Distance in m')
    
    
    ax2.set_title('Line Plot- Position Y GPS')
    ax2.plot(mean_state_estimate_1[1,0,:], 'salmon')  
    ax2.plot(true_mean_vehicle[1,0,:], c='k') 
    ax2.legend(loc = 'upper left')
    ax2.set_xlabel("Time in s")
    ax2.set_ylabel('Distance in m')
    
    ax3.set_title('Line Plot- Velocity X GPS')
    ax3.plot(mean_state_estimate_1[2,0,:], 'salmon')  
    ax3.plot(true_mean_vehicle[2,0,:], c='k') 
    ax3.legend(loc = 'upper left')
    ax3.set_xlabel("Time in s")
    ax3.set_ylabel('Velocity in m/s')
    
    ax4.set_title('Line Plot- Velocity Y GPS')
    ax4.plot(mean_state_estimate_1[3,0,:], 'salmon')  
    ax4.plot(true_mean_vehicle[3,0,:], c='k') 
    ax4.legend(loc = 'upper left')
    ax4.set_xlabel("Time in s")
    ax4.set_ylabel('Velocity in m/s')
    
    ax5.set_title('Line Plot- Acceleration X GPS')
    ax5.plot(mean_state_estimate_1[4,0,:], 'salmon')  
    ax5.plot(true_mean_vehicle[4,0,:], c='k') 
    ax5.legend(loc = 'upper left')
    ax5.set_xlabel("Time in s")
    ax5.set_ylabel('Acceleration in m/s^2')
    
    ax6.set_title('Line Plot- Acceleration Y GPS')
    ax6.plot(mean_state_estimate_1[5,0,:], 'salmon')  
    ax6.plot(true_mean_vehicle[5,0,:], c='k')  
    ax6.legend(loc = 'upper left')
    ax6.set_xlabel("Time in s")
    ax6.set_ylabel('Acceleration in m/s^2')
    
    ax7.set_title('Scatter Plot- Position X-Y GPS')
    ax7.scatter(mean_state_estimate_1[0,0,:], mean_state_estimate_1[1,0,:], marker ='x', s = 10, c= 'salmon')  # ms[0] has x position estimates
    ax7.scatter(true_mean_vehicle[0,0,:], true_mean_vehicle[1,0,:], marker ='x', s = 10, c='k')  # true_mean_vehicle has x and y coordinates
    ax7.legend(loc='upper left')
    ax7.set_xlabel('X-Distance in m')
    ax7.set_ylabel('Y-Distance in m')
    
    ax8.set_title('Scatter Plot- Velocity X-Y GPS')
    ax8.scatter(mean_state_estimate_1[2,0,:], mean_state_estimate_1[3,0,:], s = 10, c='salmon')  # mean_state_estimate_1 has x and y velocity estimates
    ax8.scatter(true_mean_vehicle[2,0,:], true_mean_vehicle[3,0,:], s = 10, c='k')  # true_mean_vehicle has x and y velocities
    ax8.legend(loc = 'upper left')
    ax8.set_xlabel('X-Velocity in m/s')
    ax8.set_ylabel('Y-Velocity in m/s')
    
    ax9.set_title('Scatter Plot- Acceleration X-Y GPS')
    ax9.scatter(mean_state_estimate_1[4,0,:], mean_state_estimate_1[5,0,:],s = 10, c= 'salmon')  # mean_state_estimate_1 has x and y velocity estimates
    ax9.scatter(true_mean_vehicle[4,0,:], true_mean_vehicle[5,0,:], s = 10, c='k')  # true_mean_vehicle has x and y velocities
    ax9.legend(loc = 'upper left')
    ax9.set_xlabel('X-Acceleration in m/s^2')
    ax9.set_ylabel('Y-Acceleration in m/s^2')
   
    #fig.legend(legend_labels, markers=legend_markers, loc='upper right')
    
    plt.tight_layout()
        
    mse_pred = np.square(
                    np.subtract(np.sqrt(np.square(true_mean_vehicle[0,:,updatesteps])+ np.square(true_mean_vehicle[1,:,updatesteps])), 
                                np.sqrt(np.square(mean_state_after_prediction[0,:,updatesteps])+ np.square(mean_state_after_prediction[1,:,updatesteps]))
                    )).mean()

    rmse_pred = np.sqrt(mse_pred)
    
    mse_est_using_gps1 = np.square(
                    np.subtract(np.sqrt(np.square(true_mean_vehicle[0,:,updatesteps])+ np.square(true_mean_vehicle[1,:,updatesteps])), 
                                np.sqrt(np.square(mean_state_after_gps1_update[0,:,updatesteps])+ np.square(mean_state_after_gps1_update[1,:,updatesteps]))
                    )).mean()
    rmse_est_using_gps1 = np.sqrt(mse_est_using_gps1)
    
    mse_est_using_gps2 = np.square(
                    np.subtract(np.sqrt(np.square(true_mean_vehicle[0,:,updatesteps])+ np.square(true_mean_vehicle[1,:,updatesteps])), 
                                np.sqrt(np.square(mean_state_after_gps2_update[0,:,updatesteps])+ np.square(mean_state_after_gps2_update[1,:,updatesteps]))
                    )).mean()
    rmse_est_using_gps2 = np.sqrt(mse_est_using_gps2)
    
    mse_est_using_gps3 = np.square(
                    np.subtract(np.sqrt(np.square(true_mean_vehicle[0,:,updatesteps])+ np.square(true_mean_vehicle[1,:,updatesteps])), 
                                np.sqrt(np.square(mean_state_after_gps3_update[0,:,updatesteps])+ np.square(mean_state_after_gps3_update[1,:,updatesteps]))
                    )).mean()
    rmse_est_using_gps3 = np.sqrt(mse_est_using_gps3)
    
    mse = np.square(
                    np.subtract(np.sqrt(np.square(true_mean_vehicle[0,:,updatesteps])+ np.square(true_mean_vehicle[1,:,updatesteps])), 
                                np.sqrt(np.square(mean_state_estimate_1[0,:,updatesteps])+ np.square(mean_state_estimate_1[1,:,updatesteps]))
                    )).mean()
    rmse = np.sqrt(mse)
    
    #return np.array([rmse_pred, rmse_est_using_gps1, rmse_est_using_gps2, rmse_est_using_gps3, rmse])
    return np.array([rmse_pred, rmse])

if __name__ == '__main__':
    plt.ion()
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Create a list of sensor names and RMSE values
    #labels = ['Prediction','Update\nGPS1', 'Update\nGPS2', 'Update\nGPS3', 'Class\nEstimate']
    labels = ['Prediction', 'Correction']
    rmse_values_50 = kalman_filter_localization(50)
    rmse_values_20 = kalman_filter_localization(20)
    
    # Plot the RMSE values
    ax1.bar(labels, rmse_values_50, capsize=5, color='tab:blue', label = 'Dataset-1')
    ax1.set_title('RMSE values Position\n measurement_rate = after 50 steps')
    ax1.set_xlabel('Mean States', fontsize=14)
    ax1.set_ylabel('RMSE (in meters)', fontsize=14)
    ax1.legend(loc = 'upper right')
    
    ax2.bar(labels, rmse_values_20, capsize=5, color='tab:green', label = 'Dataset-2')
    ax2.set_title('RMSE values Position\n measurement_rate = after 20 steps')
    ax2.set_xlabel('Mean States', fontsize=14)
    ax2.set_ylabel('RMSE (in meters)', fontsize=14)
    ax2.legend(loc = 'upper right')
    
    #plt.tight_layout()
    #fig.savefig('bar_plot_rmse_50_20.png')
    plt.show()