# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:17:45 2023

@author: Z0168020
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from UKF_4_st_2_meas import UKF
from EKF_4_st_2_meas import EKF
from LKF_4_st_2_meas import KF
from functions import convert_to_utm


num_states_ekf1 = 4
num_states_ukf = 4
measu_inputs = 2
control_inputs = 2
df = pd.read_csv("../Sort Arduino Data by Time Stamps/megadatabase_path5_14042023.csv")

df_piksi_smartphone = df[['TimeStamps','lat_piksi','lon_piksi','lat_smartphone','lon_smartphone']]
df_piksi_smartphone = df_piksi_smartphone.dropna().reset_index()
df_piksi_arduino = df[['TimeStamps','lat_piksi','lon_piksi','lat_GPS1','lon_GPS1','lat_GPS2','lon_GPS2','lat_GPS3','lon_GPS3']]
df_piksi_arduino = df_piksi_arduino.dropna().reset_index()

convert_to_utm(df_piksi_smartphone,'lat_piksi', 'lon_piksi')
convert_to_utm(df_piksi_smartphone,'lat_smartphone', 'lon_smartphone')
convert_to_utm(df_piksi_arduino,'lat_piksi', 'lon_piksi')
convert_to_utm(df_piksi_arduino,'lat_GPS1','lon_GPS1')
convert_to_utm(df_piksi_arduino,'lat_GPS2','lon_GPS2')
convert_to_utm(df_piksi_arduino,'lat_GPS3','lon_GPS3')

plt.ion()
plt.figure(figsize=(10,5))

init_state_ekf1 = np.array([681841, 5406351, 1.2, 0.005]).reshape(num_states_ekf1,1)
init_state_ukf = np.array([681970, 5406251, 1.2, 0.005]).reshape(num_states_ukf,1)

init_cov = np.eye(num_states_ekf1)

u = np.array([36.364, 
              0.005]).reshape(2,1)

process_variance = np.array([[1.5,  0,  0,     0],
                             [0,  1.5,  0,     0],
                             [0,    0,  0.0015, 0],
                             [0,    0,    0,     0.000015]])
measu_variance_ard = 0.01
measu_variance_smp = 0.02

Q = np.eye(num_states_ekf1).dot(process_variance)       
Ra = np.eye(measu_inputs).dot(measu_variance_ard)
Rs = np.eye(measu_inputs).dot(measu_variance_smp)

z = np.array([0,0]).reshape(measu_inputs,1)


ekf1 = EKF(init_state_ekf1, init_cov, z, u, Q, Rs)
ekf2 = EKF(init_state_ekf1, init_cov, z, u, Q, Ra)

ukf1 = UKF(init_state_ukf, init_cov, z, u, Q, Rs)
ukf2 = UKF(init_state_ukf, init_cov, z, u, Q, Ra)


# minimum time step (dt) used in the simulation
time_step = 1
# total number of time steps for which simulation will run
total_time_steps = max(len(df_piksi_smartphone),len(df_piksi_arduino))
 # time step at which measurement is received
measurement_rate = 1

ekf1_mean_state_estimate = np.zeros((num_states_ekf1, 1, len(df_piksi_smartphone)))
ekf2_mean_state_estimate = np.zeros((num_states_ekf1, 1, len(df_piksi_arduino)))

ukf1_mean_state_estimate = np.zeros((num_states_ukf, 1, len(df_piksi_smartphone)))
ukf2_mean_state_estimate = np.zeros((num_states_ukf, 1, len(df_piksi_arduino)))

# We consider piksi measurements as true measurements
true_mean_state_smp = np.zeros((measu_inputs, 1, len(df_piksi_smartphone)))
true_mean_state_ard = np.zeros((measu_inputs, 1, len(df_piksi_arduino)))

for step in range(total_time_steps):
        
    
    # True mean state at time t = step
    if step < len(df_piksi_smartphone):
        true_mean_state_smp[:,:,step] = np.array([[df_piksi_smartphone.loc[step,'lat_piksi_utm_easting']],
                                    [df_piksi_smartphone.loc[step,'lon_piksi_utm_northing']]])
    if step < len(df_piksi_arduino):
        true_mean_state_ard[:,:,step] = np.array([[df_piksi_arduino.loc[step,'lat_piksi_utm_easting']],
                                    [df_piksi_arduino.loc[step,'lon_piksi_utm_northing']]])

    
    if step>0:

        ekf1.predict(dt=time_step)
        ekf2.predict(dt=time_step)
        ukf1.predict(dt=time_step)
        ukf2.predict(dt=time_step)
        
    
        if step % measurement_rate == 0:
        
        
            ekf1.update(meas_value=([[df_piksi_smartphone.loc[step,'lat_smartphone_utm_easting']],
                                [df_piksi_smartphone.loc[step,'lon_smartphone_utm_northing']]]),
                                flag = 0)
        
            ukf1.update(z=([[df_piksi_smartphone.loc[step,'lat_smartphone_utm_easting']],
                                [df_piksi_smartphone.loc[step,'lon_smartphone_utm_northing']]]),
                                flag = 0)
        
            if step<len(df_piksi_arduino.loc[:,'lat_GPS1_utm_easting']):
                ekf2.update(meas_value=([[df_piksi_arduino.loc[step,'lat_GPS1_utm_easting']],
                                [df_piksi_arduino.loc[step,'lon_GPS1_utm_northing']]]),
                                flag = 1)
                ukf2.update(z=([[df_piksi_arduino.loc[step,'lat_GPS1_utm_easting']],
                                [df_piksi_arduino.loc[step,'lon_GPS1_utm_northing']]]),
                                flag = 1)
            
            if step<len(df_piksi_arduino.loc[:,'lat_GPS2_utm_easting']):
                ekf2.update(meas_value=([[df_piksi_arduino.loc[step,'lat_GPS2_utm_easting']],
                                [df_piksi_arduino.loc[step,'lon_GPS2_utm_northing']]]),
                                flag = 2)
                ukf2.update(z=([[df_piksi_arduino.loc[step,'lat_GPS2_utm_easting']],
                                [df_piksi_arduino.loc[step,'lon_GPS2_utm_northing']]]),
                                flag = 2)
            
            if step<len(df_piksi_arduino.loc[:,'lat_GPS3_utm_easting']):
                ekf2.update(meas_value=([[df_piksi_arduino.loc[step,'lat_GPS3_utm_easting']],
                                [df_piksi_arduino.loc[step,'lon_GPS3_utm_northing']]]),
                                flag = 3)
                ukf2.update(z=([[df_piksi_arduino.loc[step,'lat_GPS3_utm_easting']],
                                [df_piksi_arduino.loc[step,'lon_GPS3_utm_northing']]]),
                                flag = 3)
            
    if step < len(df_piksi_smartphone):         
        ekf1_mean_state_estimate[:,:,step] = ekf1.mean
        ukf1_mean_state_estimate[:,:,step] = ukf1.mean
    if step < len(df_piksi_arduino):
        ekf2_mean_state_estimate[:,:,step] = ekf2.mean
        ukf2_mean_state_estimate[:,:,step] = ukf2.mean

# Create a figure with two subplots
fig, ((ax1, ax2),(ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 15))
#fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))

# plt.subplot(2, 2, 1)
ax1.set_title('Position-Smartphone')
ax1.set_xlabel('Time (in s)')
ax1.set_ylabel('X-Y Position (in m)')
ax1.plot(np.sqrt(np.square(ekf1_mean_state_estimate[0,0,:])+ np.square(ekf1_mean_state_estimate[1,0,:])),'r', label = 'smartphone EKF Pos estimate')
ax1.plot(np.sqrt(np.square(ukf1_mean_state_estimate[0,0,:])+ np.square(ukf1_mean_state_estimate[1,0,:])),'g', label = 'smartphone UKF Pos Estimate')
ax1.plot(np.sqrt(np.square(true_mean_state_smp[0,0,:])+ np.square(true_mean_state_smp[1,0,:]))          ,'m', label = 'piksi Pos')
ax1.legend(loc = 'lower right')
ax1.ticklabel_format(style='plain', useOffset=False)

ax2.set_title('Position-Arduino')
ax2.set_xlabel('Time (in s)')
ax2.set_ylabel('X-Y Position (in m)')
ax2.plot(np.sqrt(np.square(ekf2_mean_state_estimate[0,0,:])+ np.square(ekf2_mean_state_estimate[1,0,:])),'r', label = '3 GPS EKF Pos estimate')
ax2.plot(np.sqrt(np.square(ukf2_mean_state_estimate[0,0,:])+ np.square(ukf2_mean_state_estimate[1,0,:])),'g', label = '3 GPS UKF Pos Estimate')
ax2.plot(np.sqrt(np.square(true_mean_state_ard[0,0,:])+ np.square(true_mean_state_ard[1,0,:]))          ,'m', label = 'piksi Pos')
ax2.legend(loc = 'lower right')
ax2.ticklabel_format(style='plain', useOffset=False)

ax3.set_title('Heading-Smartphone')
ax3.set_xlabel('Time (in s)')
ax3.set_ylabel('Heading (in radians)')
ax3.plot(ekf1_mean_state_estimate[2,0,:],'r', label = 'smartphone EKF Heading estimate')
ax3.plot(ukf1_mean_state_estimate[2,0,:],'g', label = 'smartphone UKF Heading Estimate')
ax3.legend(loc = 'upper right')
ax3.ticklabel_format(style='plain', useOffset=False)

ax4.set_title('Heading-Arduino')
ax4.set_xlabel('Time (in s)')
ax4.set_ylabel('Heading (in radians)')
ax4.plot(ekf2_mean_state_estimate[2,0,:],'r', label = '3 GPS EKF Heading estimate')
ax4.plot(ukf2_mean_state_estimate[2,0,:],'g', label = '3 GPS UKF Heading Estimate')
ax4.legend(loc = 'upper right')
ax4.ticklabel_format(style='plain', useOffset=False)


# RMSE for smartphone readings and EKF UKF
piksi_pos_smp = np.sqrt(
    np.square(np.array(df_piksi_smartphone.loc[:,'lat_piksi_utm_easting']))
    +np.square(np.array(df_piksi_smartphone.loc[:,'lon_piksi_utm_northing'])))
smartphone_sensor_pos = np.sqrt(
    np.square(np.array(df_piksi_smartphone.loc[:,'lat_smartphone_utm_easting']))
    +np.square(np.array(df_piksi_smartphone.loc[:,'lon_smartphone_utm_northing'])))
smartphone_ekf1_pos_est = np.sqrt(
    np.square(ekf1_mean_state_estimate[0,0,:])
    +np.square(ekf1_mean_state_estimate[1,0,:]))
smartphone_ukf1_pos_est = np.sqrt(
    np.square(ukf1_mean_state_estimate[0,0,:])
    +np.square(ukf1_mean_state_estimate[1,0,:]))

mse_smartphone_sensor = np.square(np.subtract(piksi_pos_smp[:], smartphone_sensor_pos[:])).mean()
mse_smartphone_ekf1 = np.square(np.subtract(piksi_pos_smp[3:],smartphone_ekf1_pos_est[3:])).mean()
mse_smartphone_ukf1 = np.square(np.subtract(piksi_pos_smp[5:],smartphone_ukf1_pos_est[5:])).mean()

rmse_smartphone_sensor = np.sqrt(mse_smartphone_sensor)
rmse_smartphone_ekf1 = np.sqrt(mse_smartphone_ekf1)
rmse_smartphone_ukf1 = np.sqrt(mse_smartphone_ukf1)

# RMSE for 3 GPS readings and EKF UKF

piksi_pos_ard = np.sqrt(
    np.square(np.array(df_piksi_arduino.loc[:,'lat_piksi_utm_easting']))
    + np.square(np.array(df_piksi_arduino.loc[:,'lon_piksi_utm_northing'])))
gps1_sensor_pos = np.sqrt(
    np.square(np.array(df_piksi_arduino.loc[:,'lat_GPS1_utm_easting']))
    + np.square(np.array(df_piksi_arduino.loc[:,'lon_GPS1_utm_northing'])))
gps2_sensor_pos = np.sqrt(
    np.square(np.array(df_piksi_arduino.loc[:,'lat_GPS2_utm_easting']))
    + np.square(np.array(df_piksi_arduino.loc[:,'lon_GPS2_utm_northing'])))
gps3_sensor_pos = np.sqrt(
    np.square(np.array(df_piksi_arduino.loc[:,'lat_GPS3_utm_easting']))
    + np.square(np.array(df_piksi_arduino.loc[:,'lon_GPS3_utm_northing'])))
all_gps_ekf2_pos_est = np.sqrt(
    np.square(ekf2_mean_state_estimate[0,0,:])
    +np.square(ekf2_mean_state_estimate[1,0,:]))
all_gps_ukf2_pos_est = np.sqrt(
    np.square(ukf2_mean_state_estimate[0,0,:])
    +np.square(ukf2_mean_state_estimate[1,0,:]))

mse_gps1_sensor = np.square(np.subtract(piksi_pos_ard[5:], gps1_sensor_pos[5:])).mean()
mse_gps2_sensor = np.square(np.subtract(piksi_pos_ard[5:], gps2_sensor_pos[5:])).mean()
mse_gps3_sensor = np.square(np.subtract(piksi_pos_ard[5:], gps3_sensor_pos[5:])).mean()
mse_all_gps_ekf2 = np.square(np.subtract(piksi_pos_ard[5:],all_gps_ekf2_pos_est[5:])).mean()
mse_all_gps_ukf2 = np.square(np.subtract(piksi_pos_ard[5:],all_gps_ukf2_pos_est[5:])).mean()

rmse_gps1_sensor = np.sqrt(mse_gps1_sensor)
rmse_gps2_sensor = np.sqrt(mse_gps2_sensor)
rmse_gps3_sensor = np.sqrt(mse_gps3_sensor)
rmse_all_gps_ekf2 = np.sqrt(mse_all_gps_ekf2)
rmse_all_gps_ukf2 = np.sqrt(mse_all_gps_ukf2)


# Create a list of smartphone sensor labels and RMSE values
labels_smp = ['Smartphone_Sensor', 'Smartphone_EKF', 'Smartphone_UKF']
values_smp = [rmse_smartphone_sensor, rmse_smartphone_ekf1, rmse_smartphone_ukf1]

# Plot the RMSE values
ax5.bar(labels_smp, values_smp, capsize=5, color='tab:blue', label = 'Dataset-1')
ax5.set_title('RMSE Error Kalman Filter')
ax5.set_xlabel('Sensor Set', fontsize=14)
ax5.set_ylabel('RMSE (in meters)', fontsize=14)
ax5.legend(loc = 'lower right')

for i in range(len(labels_smp)):
    ax5.text(i, values_smp[i]-0.15, "{:.4f}".format(values_smp[i]), ha='center', va='center', color = "white", weight='bold', fontsize=12)
    
# Create a list of GPS sensor labels and RMSE values
labels_ard = ['GPS1_Sensor', 'GPS2_Sensor', 'GPS3_Sensor', 'All_GPS_EKF', 'All_GPS_UKF']
values_ard = [rmse_gps1_sensor, rmse_gps2_sensor, rmse_gps3_sensor, rmse_all_gps_ekf2, rmse_all_gps_ukf2]

# Plot the RMSE values
ax6.bar(labels_ard, values_ard, capsize=5, color='tab:blue', label = 'Dataset-1')
ax6.set_title('RMSE Error Kalman Filter')
ax6.set_xlabel('Sensor Set', fontsize=14)
ax6.set_ylabel('RMSE (in meters)', fontsize=14)
ax6.legend(loc = 'lower right')

for i in range(len(labels_ard)):
    ax6.text(i, values_ard[i]-0.15, "{:.4f}".format(values_ard[i]), ha='center', va='center', color = "white", weight='bold', fontsize=12)     

fig.tight_layout()
plt.show()
