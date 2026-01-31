# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:18:09 2023

@author: Z0168020
"""

from UKF import UKF
from ukf_functions import convert_to_utm, calculate_rpm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians, acos


df = pd.read_csv("../Sort Arduino Data by Time Stamps/megadatabase_O_shaped_traj.csv")

df_piksi_smartphone = df[['TimeStamps','lat_piksi','lon_piksi','lat_smartphone','lon_smartphone']]
df_piksi_smartphone = df_piksi_smartphone.dropna().reset_index()
df_piksi_arduino = df[['TimeStamps','lat_piksi','lon_piksi','lat_GPS1','lon_GPS1','RPM1', 'lat_GPS2','lon_GPS2', 'RPM2','lat_GPS3','lon_GPS3','RPM3']]
df_piksi_arduino = df_piksi_arduino.dropna().reset_index()

convert_to_utm(df_piksi_smartphone,'lat_piksi', 'lon_piksi')
convert_to_utm(df_piksi_smartphone,'lat_smartphone', 'lon_smartphone')
convert_to_utm(df_piksi_arduino,'lat_piksi', 'lon_piksi')
convert_to_utm(df_piksi_arduino,'lat_GPS1','lon_GPS1')
convert_to_utm(df_piksi_arduino,'lat_GPS2','lon_GPS2')
convert_to_utm(df_piksi_arduino,'lat_GPS3','lon_GPS3')

# Apply the function to each DataFrame
calculate_rpm(df_piksi_arduino)
calculate_rpm(df_piksi_smartphone)

plt.ion()
plt.figure(figsize=(10,5))
num_states = 4
measu_inputs = 2
control_inputs = 2
rad_wheel =  0.055

init_state_ard = np.array([df_piksi_arduino.loc[0,'lat_piksi_utm_easting'], df_piksi_arduino.loc[0,'lon_piksi_utm_northing'], 1.2, 0.005]).reshape(num_states,1)
init_state_smp = np.array([df_piksi_smartphone.loc[0,'lat_piksi_utm_easting'], df_piksi_smartphone.loc[0,'lon_piksi_utm_northing'], 1.2, 0.002]).reshape(num_states,1)

init_cov = np.eye(num_states)
u = np.array([36.364, 
              0.005]).reshape(2,1)

process_variance = np.array([[1.5,  0,  0,     0],
                             [0,  1.5,  0,     0],
                             [0,    0,  0.0015, 0],
                             [0,    0,    0,     0.000015]])
measu_variance_ard = 5
measu_variance_smp = 2

Q = np.eye(num_states).dot(process_variance)       
Ra = np.eye(measu_inputs).dot(measu_variance_ard)
Rs = np.eye(measu_inputs).dot(measu_variance_smp)
z = np.array([0,0]).reshape(measu_inputs,1)


ukf1 = UKF(init_state_smp, init_cov, z, u, Q, Rs)
ukf2 = UKF(init_state_ard, init_cov, z, u, Q, Ra)

# minimum time step (dt) used in the simulation
time_step = 1
# total number of time steps for which simulation will run
total_time_steps = len(df_piksi_smartphone)
 # time step at which measurement is received
measurement_rate = 1

mean_state_estimate_1 = np.zeros((num_states, 1, len(df_piksi_smartphone)))
mean_state_estimate_2 = np.zeros((num_states, 1, len(df_piksi_arduino)))
true_mean_vehicle_1 = np.zeros(( measu_inputs, 1, len(df_piksi_smartphone)))
true_mean_vehicle_2 = np.zeros(( measu_inputs, 1, len(df_piksi_arduino)))

true_mean_vehicle_1[:,:,0] = np.array([[df_piksi_smartphone.loc[0,'lat_piksi_utm_easting']],
                            [df_piksi_smartphone.loc[0,'lon_piksi_utm_northing']]])

true_mean_vehicle_2[:,:,0] = np.array([[df_piksi_arduino.loc[0,'lat_piksi_utm_easting']],
                            [df_piksi_arduino.loc[0,'lon_piksi_utm_northing']]])

#mean_state_estimate_1 = []  #property of kalman_filter class
#mean_state_estimate_error_1 = [] #property of kalman_filter class
#mean_state_estimate_2 = []  #property of kalman_filter class
#mean_state_estimate_error_2 = [] #property of kalman_filter class
updatesteps1 = []
updatesteps2 = []
mean_state_after_prediction_ukf1 = np.zeros((num_states, 1, len(df_piksi_smartphone)))
mean_state_after_prediction_ukf2 = np.zeros((num_states, 1, len(df_piksi_arduino)))

true_mean_state_over_total_time_steps_vehicle_center = []
true_mean_state_over_total_time_steps_gps1 = []
true_mean_state_over_total_time_steps_gps2 = []
true_mean_state_over_total_time_steps_gps3 = []
#mean_state_after_prediction_ukf1 = []
#mean_state_after_prediction_ukf2 = []
true_mean_state_over_total_time_steps_smartphone= []


for step in range(total_time_steps):
          
    if step>0:
        #mean_state_estimate_1.append(ukf1.mean)
        #mean_state_estimate_2.append(ukf2.mean)
    
        # True mean state at time t = step
        true_mean_vehicle_1[:,:,step] = np.array([[df_piksi_smartphone.loc[step,'lat_piksi_utm_easting']],
                                    [df_piksi_smartphone.loc[step,'lon_piksi_utm_northing']]])
        if step<len(df_piksi_arduino):
            true_mean_vehicle_2[:,:,step] = np.array([[df_piksi_arduino.loc[step,'lat_piksi_utm_easting']],
                                    [df_piksi_arduino.loc[step,'lon_piksi_utm_northing']]])
    
        ukf1.U[0] = df_piksi_smartphone.loc[step,'RPMpiksi']* (2*np.pi/60)
        if step<len(df_piksi_arduino):
            ukf2.U[0] = df_piksi_arduino.loc[step,'RPM1']* (2*np.pi/60)
            
        ukf1.predict(dt=df_piksi_smartphone.loc[step,'deltaT'])
        if step<len(df_piksi_arduino):
            ukf2.predict(dt=df_piksi_arduino.loc[step,'deltaT'])
        
        mean_state_after_prediction_ukf1[:,:,step] = ukf1.mean
        if step<len(df_piksi_arduino):
            mean_state_after_prediction_ukf2[:,:,step] = ukf2.mean
        #mean_state_after_prediction_ukf1.append(ukf1.mean)
        #mean_state_after_prediction_ukf2.append(ukf2.mean)
        
        if step != 0 and step % measurement_rate == 0:
            updatesteps1.append(step)
            if step<len(df_piksi_arduino):
                updatesteps2.append(step)
                
            ukf1.update(z=([[df_piksi_smartphone.loc[step,'lat_smartphone_utm_easting']],
                                    [df_piksi_smartphone.loc[step,'lon_smartphone_utm_northing']]]),
                                    dt = df_piksi_smartphone.loc[step,'deltaT'],
                                    flag = 0)
            if step<len(df_piksi_arduino.loc[:,'lat_GPS1_utm_easting']) and step != 49:
                ukf2.update(z=([[df_piksi_arduino.loc[step,'lat_GPS1_utm_easting']-0],
                                    [df_piksi_arduino.loc[step,'lon_GPS1_utm_northing']-1]]),
                                    dt = df_piksi_arduino.loc[step,'deltaT'],
                                    flag = 1)
            if step<len(df_piksi_arduino.loc[:,'lat_GPS2_utm_easting']):
                ukf2.update(z=([[df_piksi_arduino.loc[step,'lat_GPS2_utm_easting']-0.8],
                                    [df_piksi_arduino.loc[step,'lon_GPS2_utm_northing']+0.8]]),
                                    dt = df_piksi_arduino.loc[step,'deltaT'],
                                    flag = 2)
            if step<len(df_piksi_arduino.loc[:,'lat_GPS3_utm_easting']):
                ukf2.update(z=([[df_piksi_arduino.loc[step,'lat_GPS3_utm_easting']+0.8],
                                    [df_piksi_arduino.loc[step,'lon_GPS3_utm_northing']+0.8]]),
                                    dt = df_piksi_arduino.loc[step,'deltaT'],
                                    flag = 3)
    
    mean_state_estimate_1[:,:,step] = ukf1.mean
    if step<len(df_piksi_arduino):
        mean_state_estimate_2[:,:,step] = ukf2.mean
# Create a figure with two subplots
fig, ((ax1, ax3)) = plt.subplots(1, 2, figsize=(10, 5))

# plt.subplot(2, 2, 1)
ax1.set_title('Position')
ax1.set_xlabel('Time (in s)')
ax1.set_ylabel('X-Y Position (in m)')
ax1.plot(mean_state_estimate_1[0,0,:], mean_state_estimate_1[1,0,:], color='indigo', linestyle='--', label = 'Smartphone Pos estimate')
ax1.plot(mean_state_estimate_2[0,0,:], mean_state_estimate_2[1,0,:], color='salmon', linestyle='--', label = 'Smartphone Pos estimate')
ax1.plot(df_piksi_smartphone.loc[:,'lat_piksi_utm_easting'], df_piksi_smartphone.loc[:,'lon_piksi_utm_northing'], 'k--', label = 'Piksi Pos')
ax1.ticklabel_format(style='plain', useOffset=False)
ax1.tick_params(axis='x', rotation=45)
ax1.legend(loc = 'upper right')

ax3.set_title('Heading')
ax3.set_xlabel('Time (in s)')
ax3.set_ylabel('Heading (in radians)')
ax3.plot(mean_state_estimate_1[2,0,:], color='indigo', linestyle='--', label = 'Smartphone Pos estimate')
ax3.plot(mean_state_estimate_2[2,0,:], color='salmon', linestyle='--', label = 'Smartphone Pos estimate')
ax3.legend(loc = 'upper right')


fig.tight_layout()
plt.show()

mse_pred_1 = np.square(
                np.subtract(np.sqrt(np.square(true_mean_vehicle_1[0,:,updatesteps1])+ np.square(true_mean_vehicle_1[1,:,updatesteps1])), 
                            np.sqrt(np.square(mean_state_after_prediction_ukf1[0,:,updatesteps1])+ np.square(mean_state_after_prediction_ukf1[1,:,updatesteps1]))
                )).mean()

rmse_pred_1 = np.sqrt(mse_pred_1)

mse_pred_2 = np.square(
                np.subtract(np.sqrt(np.square(true_mean_vehicle_2[0,:,updatesteps2])+ np.square(true_mean_vehicle_2[1,:,updatesteps2])), 
                            np.sqrt(np.square(mean_state_after_prediction_ukf2[0,:,updatesteps2])+ np.square(mean_state_after_prediction_ukf2[1,:,updatesteps2]))
                )).mean()

rmse_pred_2 = np.sqrt(mse_pred_2)


mse_1 = np.square(
                np.subtract(np.sqrt(np.square(true_mean_vehicle_1[0,:,updatesteps1])+ np.square(true_mean_vehicle_1[1,:,updatesteps1])), 
                            np.sqrt(np.square(mean_state_estimate_1[0,:,updatesteps1])+ np.square(mean_state_estimate_1[1,:,updatesteps1]))
                )).mean()
rmse_1 = np.sqrt(mse_1)

mse_2 = np.square(
                np.subtract(np.sqrt(np.square(true_mean_vehicle_2[0,:,updatesteps2])+ np.square(true_mean_vehicle_2[1,:,updatesteps2])), 
                            np.sqrt(np.square(mean_state_estimate_2[0,:,updatesteps2])+ np.square(mean_state_estimate_2[1,:,updatesteps2]))
                )).mean()
rmse_2 = np.sqrt(mse_2)

# # Calculating the distace from X and Y cordinates
# pos_piksi_after_filter_s = np.sqrt(
#     np.square(np.array(df_piksi_smartphone.loc[:,'lat_piksi_utm_easting']))
#     +np.square(np.array(df_piksi_smartphone.loc[:,'lon_piksi_utm_northing'])))

# pos_smartphone_after_filter = np.sqrt(
#     np.square(np.array([ms[0] for ms in mean_state_estimate_1]))
#     +np.square(np.array([ms[1] for ms in mean_state_estimate_1])))


# pos_piksi_after_filter_a = np.sqrt(
#     np.square(np.array(df_piksi_arduino.loc[:,'lat_piksi_utm_easting']))
#     +np.square(np.array(df_piksi_arduino.loc[:,'lon_piksi_utm_northing'])))


# pos_3gps_after_filter = np.sqrt(
#     np.square(np.array([ms[0] for ms in mean_state_estimate_2]))
#     +np.square(np.array([ms[1] for ms in mean_state_estimate_2]))) 

# # RMSE for smartphone
# mse_smartphone_after_filter = np.square(np.subtract(pos_piksi_after_filter_s[:],pos_smartphone_after_filter[:])).mean()
# rmse_smartphone_after_filter = np.sqrt(mse_smartphone_after_filter)
# # RMSE for arduino    
# mse_est_3gps_after_filter = np.square(np.subtract(pos_piksi_after_filter_a[:], pos_3gps_after_filter[:])).mean()
# rmse_est_3gps_after_filter = np.sqrt(mse_est_3gps_after_filter)


# Create a figure with one subplot
fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))

# Create a list of sensor names and RMSE values
labels = ['Prediction', 'Class\nEstimate']
rmse_values = np.array([[rmse_pred_1, rmse_pred_2], [rmse_1, rmse_2]])

# Transpose the rmse_values array
rmse_values = rmse_values.T

# Plot the RMSE values
x = np.arange(len(labels))  # x-coordinates for the bars
width = 0.35  # width of the bars
colors = ['tab:blue', 'tab:orange']  # colors for the bars
for i in range(rmse_values.shape[1]):
    ax2.bar(x + i * width, rmse_values[:, i], width, label='Dataset-{}'.format(i+1), color=colors[i])

ax2.set_title('RMSE values Position X-Y GPS\nmeasurement_rate = after 50 steps')
ax2.set_xlabel('Mean States', fontsize=14)
ax2.set_ylabel('RMSE (in meters)', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend(loc='upper right')

fig2.tight_layout()
plt.show()