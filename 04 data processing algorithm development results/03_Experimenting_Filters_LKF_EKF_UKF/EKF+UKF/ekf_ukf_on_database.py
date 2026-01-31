# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:15:34 2023

@author: Z0168020
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:18:09 2023

@author: Z0168020
"""

from UKF import UKF
from EKF import EKF
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


ukf2 = UKF(init_state_ard, init_cov, z, u, Q, Ra)
ekf2 = EKF(init_state_ard, init_cov, z, u, Q, Ra)

# minimum time step (dt) , total number of time step, time step 
time_step, total_time_steps, measurement_rate = 1, max(len(df_piksi_smartphone), len(df_piksi_arduino)), 1 

mean_state_estimate_ukf2 = np.zeros((num_states, 1, len(df_piksi_arduino)))
mean_state_estimate_ekf2 = np.zeros((num_states, 1, len(df_piksi_arduino)))

true_mean_vehicle_1 = np.zeros(( measu_inputs, 1, len(df_piksi_smartphone)))
true_mean_vehicle_2 = np.zeros(( measu_inputs, 1, len(df_piksi_arduino)))

true_mean_vehicle_1[:,:,0] = np.array([[df_piksi_smartphone.loc[0,'lat_piksi_utm_easting']],
                            [df_piksi_smartphone.loc[0,'lon_piksi_utm_northing']]])

true_mean_vehicle_2[:,:,0] = np.array([[df_piksi_arduino.loc[0,'lat_piksi_utm_easting']],
                            [df_piksi_arduino.loc[0,'lon_piksi_utm_northing']]])

updatesteps1 = []
updatesteps2 = []
mean_state_after_prediction_ekf2 = np.zeros((num_states, 1, len(df_piksi_arduino)))
mean_state_after_prediction_ukf2 = np.zeros((num_states, 1, len(df_piksi_arduino)))

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
    
       
        if step<len(df_piksi_arduino):
            ukf2.U[0] = df_piksi_arduino.loc[step,'RPM1']* (2*np.pi/60)
            ekf2._u[0] = df_piksi_arduino.loc[step,'RPM1']* (2*np.pi/60)
            
        
        if step<len(df_piksi_arduino):
            ekf2.predict(dt=df_piksi_arduino.loc[step,'deltaT'])
            ukf2.predict(dt=df_piksi_arduino.loc[step,'deltaT'])
            
        
        #mean_state_after_prediction_ukf1[:,:,step] = ukf1.mean
        
        if step<len(df_piksi_arduino):
            mean_state_after_prediction_ekf2[:,:,step] = ekf2.mean
            mean_state_after_prediction_ukf2[:,:,step] = ukf2.mean
            
        #mean_state_after_prediction_ukf1.append(ukf1.mean)
        #mean_state_after_prediction_ukf2.append(ukf2.mean)
        
        if step != 0 and step % measurement_rate == 0:
            updatesteps1.append(step)
            if step<len(df_piksi_arduino):
                updatesteps2.append(step)
                
            # ukf1.update(z=([[df_piksi_smartphone.loc[step,'lat_smartphone_utm_easting']],
            #                         [df_piksi_smartphone.loc[step,'lon_smartphone_utm_northing']]]),
            #                         dt = df_piksi_smartphone.loc[step,'deltaT'],
            #                         flag = 0)
            if step<len(df_piksi_arduino.loc[:,'lat_GPS1_utm_easting']):
                ekf2.update(meas_value=([[df_piksi_arduino.loc[step,'lat_GPS1_utm_easting']-0],
                                    [df_piksi_arduino.loc[step,'lon_GPS1_utm_northing']-1]]),
                                    flag = 1)
                ukf2.update(z=([[df_piksi_arduino.loc[step,'lat_GPS1_utm_easting']-0],
                                    [df_piksi_arduino.loc[step,'lon_GPS1_utm_northing']-1]]),
                                    dt = df_piksi_arduino.loc[step,'deltaT'],
                                    flag = 1)
            if step<len(df_piksi_arduino.loc[:,'lat_GPS2_utm_easting']):
                ekf2.update(meas_value=([[df_piksi_arduino.loc[step,'lat_GPS2_utm_easting']-0.8],
                                    [df_piksi_arduino.loc[step,'lon_GPS2_utm_northing']+0.8]]),
                                    flag = 2)
                ukf2.update(z=([[df_piksi_arduino.loc[step,'lat_GPS2_utm_easting']-0.8],
                                    [df_piksi_arduino.loc[step,'lon_GPS2_utm_northing']+0.8]]),
                                    dt = df_piksi_arduino.loc[step,'deltaT'],
                                    flag = 2)
            if step<len(df_piksi_arduino.loc[:,'lat_GPS3_utm_easting']):
                ekf2.update(meas_value=([[df_piksi_arduino.loc[step,'lat_GPS3_utm_easting']+0.8],
                                    [df_piksi_arduino.loc[step,'lon_GPS3_utm_northing']+0.8]]),
                                    flag = 3)
                ukf2.update(z=([[df_piksi_arduino.loc[step,'lat_GPS3_utm_easting']+0.8],
                                    [df_piksi_arduino.loc[step,'lon_GPS3_utm_northing']+0.8]]),
                                    dt = df_piksi_arduino.loc[step,'deltaT'],
                                    flag = 3)
    
    #mean_state_estimate_1[:,:,step] = ukf1.mean
    if step<len(df_piksi_arduino):
        mean_state_estimate_ukf2[:,:,step] = ukf2.mean
        mean_state_estimate_ekf2[:,:,step] = ekf2.mean 
# Create a figure with two subplots
fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(13.5, 4.5))

# plt.subplot(2, 2, 1)
ax1.set_title('Position')
ax1.set_xlabel('X Position (in m)')
ax1.set_ylabel('Y Position (in m)')
ax1.plot(mean_state_estimate_ukf2[0,0,:], mean_state_estimate_ukf2[1,0,:], color='blue', linestyle='--', label = 'UKF Pos estimate')
ax1.plot(mean_state_estimate_ekf2[0,0,:], mean_state_estimate_ekf2[1,0,:], color='salmon', linestyle='--', label = 'EKF Pos estimate')
ax1.plot(df_piksi_arduino.loc[:,'lat_piksi_utm_easting'], df_piksi_arduino.loc[:,'lon_piksi_utm_northing'], 'k--', label = 'Piksi Pos')
ax1.plot(df_piksi_arduino.loc[:,'lat_GPS1_utm_easting'], df_piksi_arduino.loc[:,'lon_GPS1_utm_northing'], 'g--', label = 'GPS1 Pos')
ax1.plot(df_piksi_arduino.loc[:,'lat_GPS2_utm_easting'], df_piksi_arduino.loc[:,'lon_GPS2_utm_northing'], color='teal', linestyle='--', label = 'GPS2 Pos')
ax1.plot(df_piksi_arduino.loc[:,'lat_GPS3_utm_easting'], df_piksi_arduino.loc[:,'lon_GPS3_utm_northing'], color=(0, 128/255, 128/255), linestyle='--', label = 'GPS3 Pos')
#ax1.plot(df_piksi_smartphone.loc[:,'lat_smartphone_utm_easting'], df_piksi_smartphone.loc[:,'lon_smartphone_utm_northing'], color='purple', linestyle='--', label = 'Smartphone Pos')
ax1.ticklabel_format(style='plain', useOffset=False)
ax1.tick_params(axis='x', rotation=45)
ax1.legend(loc = 'lower right')
#ax1.legend(loc = 'upper right', bbox_to_anchor=(1, 1.5))

ax2.set_title('Heading')
ax2.set_xlabel('Time (in s)')
ax2.set_ylabel('Heading (in radians)')
ax2.plot(mean_state_estimate_ukf2[2,0,:], color='blue', linestyle='--', label = 'UKF heading estimate')
ax2.plot(mean_state_estimate_ekf2[2,0,:], color='salmon', linestyle='--', label = 'EKF heading estimate')
ax2.legend(loc = 'upper center')
#ax2.legend(loc = 'upper right', bbox_to_anchor=(1, 1.5))

ax3.set_title('Heading Chnage')
ax3.set_xlabel('Time (in s)')
ax3.set_ylabel('Heading change (in radians/sec)')
ax3.plot(mean_state_estimate_ukf2[3,0,:], color='blue', linestyle='--', label = 'UKF heading change estimate')
ax3.plot(mean_state_estimate_ekf2[3,0,:], color='salmon', linestyle='--', label = 'EKF heading chnage estimate')
ax3.legend(loc = 'lower right')
#ax3.legend(loc = 'upper right', bbox_to_anchor=(1, 1.5))


fig.tight_layout()
plt.show()

mse_pred_ukf2 = np.square(
                np.subtract(np.sqrt(np.square(true_mean_vehicle_2[0,:,updatesteps2])+ np.square(true_mean_vehicle_2[1,:,updatesteps2])), 
                            np.sqrt(np.square(mean_state_after_prediction_ukf2[0,:,updatesteps2])+ np.square(mean_state_after_prediction_ukf2[1,:,updatesteps2]))
                )).mean()

rmse_pred_ukf2 = np.sqrt(mse_pred_ukf2)

mse_pred_ekf2 = np.square(
                np.subtract(np.sqrt(np.square(true_mean_vehicle_2[0,:,updatesteps2])+ np.square(true_mean_vehicle_2[1,:,updatesteps2])), 
                            np.sqrt(np.square(mean_state_after_prediction_ekf2[0,:,updatesteps2])+ np.square(mean_state_after_prediction_ekf2[1,:,updatesteps2]))
                )).mean()

rmse_pred_ekf2 = np.sqrt(mse_pred_ekf2)


mse_ukf2 = np.square(
                np.subtract(np.sqrt(np.square(true_mean_vehicle_2[0,:,updatesteps2])+ np.square(true_mean_vehicle_2[1,:,updatesteps2])), 
                            np.sqrt(np.square(mean_state_estimate_ukf2[0,:,updatesteps2])+ np.square(mean_state_estimate_ukf2[1,:,updatesteps2]))
                )).mean()
rmse_ukf2 = np.sqrt(mse_ukf2)

mse_ekf2 = np.square(
                np.subtract(np.sqrt(np.square(true_mean_vehicle_2[0,:,updatesteps2])+ np.square(true_mean_vehicle_2[1,:,updatesteps2])), 
                            np.sqrt(np.square(mean_state_estimate_ekf2[0,:,updatesteps2])+ np.square(mean_state_estimate_ekf2[1,:,updatesteps2]))
                )).mean()
rmse_ekf2 = np.sqrt(mse_ekf2)


mse_smartphone = np.square(
                    np.subtract(np.sqrt(np.square(df_piksi_smartphone.loc[:,'lat_piksi_utm_easting'])+np.square(df_piksi_smartphone.loc[:,'lon_piksi_utm_northing'])),
                                np.sqrt(np.square(df_piksi_smartphone.loc[:,'lat_smartphone_utm_easting'])+np.square(df_piksi_smartphone.loc[:,'lon_smartphone_utm_northing']))
                )).mean()

rmse_smartphone = np.sqrt(mse_smartphone)

mse_gps1 = np.square(
                    np.subtract(np.sqrt(np.square(df_piksi_arduino.loc[:,'lat_piksi_utm_easting'])+np.square(df_piksi_arduino.loc[:,'lon_piksi_utm_northing'])),
                                np.sqrt(np.square(df_piksi_arduino.loc[:,'lat_GPS1_utm_easting'])+np.square(df_piksi_arduino.loc[:,'lon_GPS1_utm_northing']))
                )).mean()

rmse_gps1 = np.sqrt(mse_gps1)
mse_gps2 = np.square(
                    np.subtract(np.sqrt(np.square(df_piksi_arduino.loc[:,'lat_piksi_utm_easting'])+np.square(df_piksi_arduino.loc[:,'lon_piksi_utm_northing'])),
                                np.sqrt(np.square(df_piksi_arduino.loc[:,'lat_GPS2_utm_easting'])+np.square(df_piksi_arduino.loc[:,'lon_GPS2_utm_northing']))
                )).mean()

rmse_gps2 = np.sqrt(mse_gps2)
mse_gps3 = np.square(
                    np.subtract(np.sqrt(np.square(df_piksi_arduino.loc[:,'lat_piksi_utm_easting'])+np.square(df_piksi_arduino.loc[:,'lon_piksi_utm_northing'])),
                                np.sqrt(np.square(df_piksi_arduino.loc[:,'lat_GPS3_utm_easting'])+np.square(df_piksi_arduino.loc[:,'lon_GPS3_utm_northing']))
                )).mean()

rmse_gps3 = np.sqrt(mse_gps3)

fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))

# Create a list of sensor names and RMSE values
labels = ['Prediction UKF', 'Prediction EKF', 'Correction UKF', 'Correction EKF', 'Raw Smartphone', 'GPS1', 'GPS2', 'GPS3']
rmse_values = np.array([rmse_pred_ukf2, rmse_pred_ekf2, rmse_ukf2, rmse_ekf2, rmse_smartphone, rmse_gps1, rmse_gps2, rmse_gps3])

# Plot the RMSE values
x = np.arange(len(labels))  # x-coordinates for the bars
width = 0.35  # width of the bars
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:green', 'tab:green', 'tab:green']  # colors for the bars
# for i, val in enumerate(rmse_values):
#     ax2.bar(x + i * width, val, width, label=labels[i], color=colors[i])
#     #ax2.text(x + i * width, val, "{:.4f}".format(val), ha='center', va='bottom')
#ax2.bar(x , rmse_values, width, label=labels, color=colors)
bars = ax2.bar(x, rmse_values, width, color=colors)
ax2.set_title('RMSE values Position X-Y')
ax2.set_xlabel('Mean States', fontsize=14)
ax2.set_ylabel('RMSE (in meters)', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right')
ax2.legend(loc='upper right')

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, height, '{:.4f}'.format(height),
            ha='center', va='bottom')

plt.show()

# # Create a figure with one subplot
# fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))

# # Create a list of sensor names and RMSE values
# labels = ['Prediction UKF', 'Prediction EKF', 'Correction UKF', 'Correction EKF', 'Raw Smartphone']
# rmse_values = np.array([rmse_pred_ukf2, rmse_pred_ekf2, rmse_ukf2, rmse_ekf2, rmse_smartphone])

# # Transpose the rmse_values array
# rmse_values = rmse_values.T

# # Plot the RMSE values
# x = np.arange(len(labels))  # x-coordinates for the bars
# width = 0.35  # width of the bars
# colors = ['tab:blue', 'tab:orange']  # colors for the bars
# for i in range(rmse_values.shape[0]):
#     ax2.bar(x + i * width, rmse_values[i], width, label=labels[i], color=colors[i])
#     for j, val in enumerate(rmse_values[i, :]):
#         ax2.text(x[i] + i * width, val, "{:.4f}".format(val), ha='center', va='top')


# ax2.set_title('RMSE values Position X-Y')
# ax2.set_xlabel('Mean States', fontsize=14)
# ax2.set_ylabel('RMSE (in meters)', fontsize=14)
# ax2.set_xticks(x)
# ax2.set_xticklabels(labels)
# ax2.legend(loc='upper right')

# fig2.tight_layout()
# plt.show()