# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:02:18 2023

@author: Z0168020
"""
from KF2 import KF
from kf_functions import convert_to_utm

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# offsets of each variable in the mean state matrix
num_states = 6
measu_inputs = 2
control_inputs = 2
rad_wheel =  0.055
state_Dx, state_Dy, state_Vx, state_Vy, state_Ax, state_Ay = 0,1,2,3,4,5

df = pd.read_csv("../Sort Arduino Data by Time Stamps/megadatabase_linear_traj.csv")

df_piksi_smartphone = df[['TimeStamps','lat_piksi','lon_piksi','lat_smartphone','lon_smartphone']]
df_piksi_smartphone = df_piksi_smartphone.dropna().reset_index()
convert_to_utm(df_piksi_smartphone,'lat_piksi', 'lon_piksi')
convert_to_utm(df_piksi_smartphone,'lat_smartphone', 'lon_smartphone')

df_piksi_arduino = df[['TimeStamps','lat_piksi','lon_piksi','lat_GPS1','lon_GPS1','lat_GPS2','lon_GPS2','lat_GPS3','lon_GPS3']]
df_piksi_arduino = df_piksi_arduino.dropna().reset_index()
convert_to_utm(df_piksi_arduino,'lat_piksi', 'lon_piksi')
convert_to_utm(df_piksi_arduino,'lat_GPS1','lon_GPS1')
convert_to_utm(df_piksi_arduino,'lat_GPS2','lon_GPS2')
convert_to_utm(df_piksi_arduino,'lat_GPS3','lon_GPS3')


plt.ion()
plt.figure(figsize=(10,5))
num_states = 6
measu_inputs = 2
control_inputs = 2
rad_wheel =  0.055

init_state_ard = np.array([df_piksi_arduino.loc[0,'lat_piksi_utm_easting'], df_piksi_arduino.loc[1,'lon_piksi_utm_northing'], 1, 1,0.01,0.01]).reshape(num_states,1)
init_state_smp = np.array([df_piksi_smartphone.loc[0,'lat_piksi_utm_easting'], df_piksi_smartphone.loc[1,'lon_piksi_utm_northing'], 1, 1,0.01,0.01]).reshape(num_states,1)
init_cov = np.eye(num_states)
u = np.array([0.00,0.00]).reshape(2,1)
process_variance = 0.03
Q = np.eye(measu_inputs).dot(process_variance) 
sigma_measurement_arduino = 0.5
R_ard = np.eye(measu_inputs).dot(sigma_measurement_arduino)
sigma_measurement_smartphone = 0.6
R_smp = np.eye(measu_inputs).dot(sigma_measurement_smartphone)
z = np.array([0,0]).reshape(measu_inputs,1)    


kf1 = KF(init_state_ard, init_cov, z, u, Q, R_smp)
kf2 = KF(init_state_smp, init_cov, z, u, Q, R_ard)


time_step = 1
total_timesteps = max(len(df_piksi_smartphone), len(df_piksi_arduino))
measurement_rate = 1

mean_state_estimate_1 = np.zeros((num_states, 1, len(df_piksi_smartphone)))
mean_state_estimate_2 = np.zeros((num_states, 1, len(df_piksi_arduino)))
true_mean_vehicle_1 = np.zeros(( measu_inputs, 1, len(df_piksi_smartphone)))
true_mean_vehicle_2 = np.zeros(( measu_inputs, 1, len(df_piksi_arduino)))

true_mean_vehicle_1[:,:,0] = np.array([[df_piksi_smartphone.loc[0,'lat_piksi_utm_easting']],
                            [df_piksi_smartphone.loc[0,'lon_piksi_utm_northing']]])

true_mean_vehicle_2[:,:,0] = np.array([[df_piksi_arduino.loc[0,'lat_piksi_utm_easting']],
                            [df_piksi_arduino.loc[0,'lon_piksi_utm_northing']]])

updatesteps1 = []
updatesteps2 = []
mean_after_prediction_1 = np.zeros((num_states, 1, len(df_piksi_smartphone)))
mean_after_prediction_2 = np.zeros((num_states, 1, len(df_piksi_arduino)))   
    

for step in range(total_timesteps):
          
    if step >0:

        # True mean state at time t = step
        true_mean_vehicle_1[:,:,step] = np.array([[df_piksi_smartphone.loc[step,'lat_piksi_utm_easting']],
                                    [df_piksi_smartphone.loc[step,'lon_piksi_utm_northing']]])
        if step<len(df_piksi_arduino):
            true_mean_vehicle_2[:,:,step] = np.array([[df_piksi_arduino.loc[step,'lat_piksi_utm_easting']],
                                    [df_piksi_arduino.loc[step,'lon_piksi_utm_northing']]])
    
        kf1.predict(dt=time_step)
        kf2.predict(dt=time_step)
        
        mean_after_prediction_1[:,:,step] = kf1.mean
        if step<len(df_piksi_arduino):
            mean_after_prediction_2[:,:,step] = kf2.mean
        
        if step != 0 and step % measurement_rate == 0:
            
            updatesteps1.append(step)
            if step<len(df_piksi_arduino):
                updatesteps2.append(step)
            
            kf1.update(meas_value=([[df_piksi_smartphone.loc[step,'lat_smartphone_utm_easting']],
                                    [df_piksi_smartphone.loc[step,'lon_smartphone_utm_northing']]]))
            #if step<len(df_piksi_arduino.loc[:,'lat_GPS1_utm_easting']):
            if step<len(df_piksi_arduino):
                kf2.update(meas_value=([[df_piksi_arduino.loc[step,'lat_GPS1_utm_easting']],
                                    [df_piksi_arduino.loc[step,'lon_GPS1_utm_northing']]]))
            #if step<len(df_piksi_arduino.loc[:,'lat_GPS2_utm_easting']):
            if step<len(df_piksi_arduino):
                kf2.update(meas_value=([[df_piksi_arduino.loc[step,'lat_GPS2_utm_easting']],
                                    [df_piksi_arduino.loc[step,'lon_GPS2_utm_northing']]]))
            #if step<len(df_piksi_arduino.loc[:,'lat_GPS3_utm_easting']):
            if step<len(df_piksi_arduino):
                kf2.update(meas_value=([[df_piksi_arduino.loc[step,'lat_GPS3_utm_easting']],
                                    [df_piksi_arduino.loc[step,'lon_GPS3_utm_northing']]]))

     
    mean_state_estimate_1[:,:,step] = kf1.mean
    if step<len(df_piksi_arduino):
        mean_state_estimate_2[:,:,step] = kf2.mean

mse_pred_1 = np.square(
                np.subtract(np.sqrt(np.square(true_mean_vehicle_1[0,:,updatesteps1])+ np.square(true_mean_vehicle_1[1,:,updatesteps1])), 
                            np.sqrt(np.square(mean_after_prediction_1[0,:,updatesteps1])+ np.square(mean_after_prediction_1[1,:,updatesteps1]))
                )).mean()

rmse_pred_1 = np.sqrt(mse_pred_1)

mse_pred_2 = np.square(
                np.subtract(np.sqrt(np.square(true_mean_vehicle_2[0,:,updatesteps2])+ np.square(true_mean_vehicle_2[1,:,updatesteps2])), 
                            np.sqrt(np.square(mean_after_prediction_2[0,:,updatesteps2])+ np.square(mean_after_prediction_2[1,:,updatesteps2]))
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


# Create a figure with two subplots
fig, ((ax1, ax2),(ax3, ax4),(ax5, ax6)) = plt.subplots(3, 2, figsize=(10, 15))
#, ((ax5, ax6)) = plt.subplots(1, 2, figsize=(10, 5))
ax1.set_title('Position')
ax1.set_xlabel('Time (in s)')
ax1.set_ylabel('X Position (in m)')
ax1.plot(mean_state_estimate_1[0,0,:], color='cyan', linestyle='--', label = 'Smartphone X Pos estimate')
ax1.plot(mean_state_estimate_2[0,0,:], color='salmon', linestyle='--', label = 'Arduino X Pos estimate')
ax1.plot(df_piksi_smartphone.loc[:,'lat_piksi_utm_easting'], 'k--', label = 'Piksi X Pos')
ax1.ticklabel_format(style='plain', useOffset=False)
ax1.legend(loc = 'lower right')


ax3.set_title('Position')
ax3.set_xlabel('Time (in s)')
ax3.set_ylabel('Y Position (in m)')
ax3.plot(mean_state_estimate_1[1,0,:], color='cyan', linestyle='--', label = 'Smartphone Y Pos estimate')
ax3.plot(mean_state_estimate_2[1,0,:], color='salmon', linestyle='--', label = 'Arduino Y Pos estimate')
ax3.plot(df_piksi_smartphone.loc[:,'lon_piksi_utm_northing'], 'k--', label = 'Piksi Y Pos')
ax3.ticklabel_format(style='plain', useOffset=False)
ax3.legend(loc = 'upper right')

ax2.set_title('Velocity')
ax2.set_xlabel('Time (in s)')
ax2.set_ylabel('X Velocity (in m/s)')
ax2.plot(mean_state_estimate_1[2,0,:], color='cyan', linestyle='--', label = 'Smartphone X Vel estimate')
ax2.plot(mean_state_estimate_2[2,0,:], color='salmon', linestyle='--', label = 'Arduino X Vel estimate')
ax2.legend(loc = 'upper right')

#plt.subplot(2, 2, 2)
ax4.set_title('Velocity')
ax4.set_xlabel('Time (in s)')
ax4.set_ylabel('Y Velocity (in m/s)')
ax4.plot( mean_state_estimate_1[3,0,:], color='cyan', linestyle='--', label = 'Smartphone Y Vel estimate')
ax4.plot(mean_state_estimate_2[3,0,:], color='salmon', linestyle='--', label = 'Arduino Y Vel estimate')
ax4.legend(loc = 'upper right')

#plt.subplot(2, 2, 1)
ax5.set_title('Position')
ax5.set_xlabel('Time (in s)')
ax5.set_ylabel('X-Y Position (in m)')
# ax5.scatter(mean_state_estimate_1[0,0,:], mean_state_estimate_1[1,0,:], color='indigo',marker='o', label = 'Smartphone Pos estimate')
# ax5.scatter(mean_state_estimate_2[0,0,:], mean_state_estimate_2[1,0,:], color='salmon', marker='o', label = 'Smartphone Pos estimate')
# ax5.scatter(df_piksi_smartphone.loc[:,'lat_piksi_utm_easting'], df_piksi_smartphone.loc[:,'lon_piksi_utm_northing'], c='k', marker= 'o', label = 'Piksi Pos')
ax5.plot(mean_state_estimate_1[0,0,:], mean_state_estimate_1[1,0,:], color='cyan', linestyle="--", label = 'Smartphone Pos estimate')
ax5.plot(mean_state_estimate_2[0,0,:], mean_state_estimate_2[1,0,:], color='salmon', linestyle='--', label = 'Arduino Pos estimate')
ax5.plot(df_piksi_smartphone.loc[:,'lat_piksi_utm_easting'], df_piksi_smartphone.loc[:,'lon_piksi_utm_northing'], c='k', linestyle='--', label = 'Piksi Pos')
ax5.plot(df_piksi_arduino.loc[:,'lat_GPS1_utm_easting'], df_piksi_arduino.loc[:,'lon_GPS1_utm_northing'], 'g--', label = 'GPS1 Pos')
ax5.plot(df_piksi_arduino.loc[:,'lat_GPS2_utm_easting'], df_piksi_arduino.loc[:,'lon_GPS2_utm_northing'], color='teal', linestyle='--', label = 'GPS2 Pos')
ax5.plot(df_piksi_arduino.loc[:,'lat_GPS3_utm_easting'], df_piksi_arduino.loc[:,'lon_GPS3_utm_northing'], color=(0, 128/255, 128/255), linestyle='--', label = 'GPS3 Pos')
ax5.plot(df_piksi_smartphone.loc[:,'lat_smartphone_utm_easting'], df_piksi_smartphone.loc[:,'lon_smartphone_utm_northing'], color='blue', linestyle='--', label = 'GPS3 Pos')
#ax1.plot(df_piksi_smartphone.loc[:,'lat_smartphone_utm_easting'], 'g', label = 'sensor reading')
ax5.ticklabel_format(style='plain', useOffset=False)
ax5.tick_params(axis='x', rotation=45)
ax5.legend(loc = 'upper right', fontsize= 8)

#plt.subplot(2, 2, 2)
ax6.set_title('Velocity')
ax6.set_xlabel('X Velocity (in m/s)')
ax6.set_ylabel('Y Velocity (in m/s)')
ax6.scatter(mean_state_estimate_1[2,0,:], mean_state_estimate_1[3,0,:], color='cyan', marker='x', label = 'Smartphone Vel estimate')
ax6.scatter(mean_state_estimate_2[2,0,:], mean_state_estimate_2[3,0,:], color='salmon', marker='x', label = 'Smartphone Vel estimate')
ax6.legend(loc = 'upper right')

import matplotlib.pyplot as plt
import numpy as np

fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))

# Create a list of sensor names and RMSE values
labels = ['Prediction Smartphone', 'PredictionArduino Multi', 'Correction Smartphone', 'Correction Arduino Multi', 'Raw Smartphone', 'GPS1', 'GPS2', 'GPS3']
rmse_values = np.array([rmse_pred_1, rmse_pred_2, rmse_1, rmse_2, rmse_smartphone, rmse_gps1, rmse_gps2, rmse_gps3])

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
# fig2, ax8 = plt.subplots(1, 1, figsize=(5, 5))

# # Create a list of sensor names and RMSE values
# labels = ['Raw Smartphone', 'Raw Arduino Multi', 'Prediction', 'Correction']
# rmse_values = np.array([[rmse_smartphone],[rmse_gps1, rmse_gps2, rmse_gps3],[rmse_pred_1, rmse_pred_2], [rmse_1, rmse_2]])

# # Transpose the rmse_values array
# rmse_values = rmse_values.T

# # Plot the RMSE values
# x = np.arange(len(labels))  # x-coordinates for the bars
# width = 0.35  # width of the bars
# colors1 = ['tab:blue', ]  # colors for the bars
# sub_labels1 = [' Raw smartphone']
# colors2 = ['tab:green', 'tab:green', 'tab:green']  # colors for the bars
# sub_labels2 = ['GPS1', 'GPS2', 'GPS3']
# colors34 = ['tab:blue', 'tab:orange']  # colors for the bars
# sub_labels34 = ['Smartphone', 'Arduino Multi GNSS']
# for i in range(rmse_values.shape[1]):
#     if i==0:
#         ax8.bar(x + i * width, rmse_values[:, i], width, label=sub_labels1[i], color=colors1[i])
#     if i==1:
#         ax8.bar(x + i * width, rmse_values[:, i], width, label=sub_labels2[i], color=colors2[i])
#     else:
#         ax8.bar(x + i * width, rmse_values[:, i], width, label=sub_labels34[i], color=colors34[i])
#     for j, val in enumerate(rmse_values[:, i]):
#        ax8.text(x[j] + i * width, val, "{:.4f}".format(val), ha='center', va='top')
       
# ax8.set_title('RMSE values Position X-Y')
# ax8.set_xlabel('Mean States', fontsize=14)
# ax8.set_ylabel('RMSE (in meters)', fontsize=14)
# ax8.set_xticks(x + width/2)
# ax8.set_xticklabels(labels)
# ax8.legend(loc='upper right')
# ax8.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

# fig2.tight_layout()
# plt.show()
