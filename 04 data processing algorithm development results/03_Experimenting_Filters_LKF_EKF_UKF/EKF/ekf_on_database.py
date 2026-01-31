# -*- coding: utf-8 -*-
"""
Created on Mon May 29 08:53:26 2023

@author: Z0168020
"""

from EKF import EKF
from functions_joseph import convert_to_utm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_rpm(df):
    cum_diff_lat = df['lat_piksi_utm_easting'].diff()
    cum_diff_lon = df['lon_piksi_utm_northing'].diff()
    dist = np.sqrt(cum_diff_lat**2 + cum_diff_lon**2)
    cum_diff_time = pd.to_datetime(df['TimeStamps']).diff()
    ang_dist = dist / (2*np.pi*0.055)
    cum_diff_time_s = cum_diff_time.dt.total_seconds()
    ang_vel = ang_dist / cum_diff_time_s
    ang_vel_rpm = ang_vel * 60
    df['RPMpiksi'] = ang_vel_rpm
    df['RPMpiksi'] = df['RPMpiksi'].fillna(method='ffill')

df = pd.read_csv("../Sort Arduino Data by Time Stamps/megadatabase_linear_traj.csv")
#df = pd.read_csv("../Sort Arduino Data by Time Stamps/megadatabase.csv")
#df = pd.read_csv("../Sort Arduino Data by Time Stamps/megadatabase_path5_14042023.csv")

df_piksi_smartphone = df[['TimeStamps','lat_piksi','lon_piksi','lat_smartphone','lon_smartphone']]
df_piksi_smartphone = df_piksi_smartphone.dropna().reset_index()
df_piksi_arduino = df[['TimeStamps','lat_piksi','lon_piksi','lat_GPS1','lon_GPS1','RPM1','lat_GPS2','lon_GPS2','RPM2','lat_GPS3','lon_GPS3','RPM3']]
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
#plt.figure(figsize=(10,5))
num_states = 4
measu_inputs = 2
control_inputs = 2
rad_wheel =  0.055



init_state_ard = np.array([df_piksi_arduino.loc[0,'lat_piksi_utm_easting'], df_piksi_arduino.loc[0,'lon_piksi_utm_northing'], 1.2, 0.005]).reshape(num_states,1)
init_state_smp = np.array([df_piksi_smartphone.loc[0,'lat_piksi_utm_easting'], df_piksi_smartphone.loc[0,'lon_piksi_utm_northing'], 1.2, 0.002]).reshape(num_states,1)


init_cov = np.eye(num_states)
u = np.array([36.36, 
              0.005]).reshape(2,1)

process_variance = np.array([[1.5,  0,  0,     0],
                              [0,  1.5,  0,     0],
                              [0,    0,  0.0015, 0],
                              [0,    0,    0,     0.000015]])

measu_variance_ard = 0.5
measu_variance_smp = 0.2

Q = np.eye(num_states).dot(process_variance)       
Ra = np.eye(measu_inputs).dot(measu_variance_ard)
Rs = np.eye(measu_inputs).dot(measu_variance_smp)

z = np.array([0,0]).reshape(measu_inputs,1)


ekf1 = EKF(init_state_smp, init_cov, z, u, Q, Rs)

ekf2 = EKF(init_state_ard, init_cov, z, u, Q, Ra)

# minimum time step (dt) used in the simulation
time_step = 1
# total number of time steps for which simulation will run
total_time_steps = len(df_piksi_smartphone)
 # time step at which measurement is received
measurement_rate = 1


mean_state_estimate_1 = []  #property of kalman_filter class
mean_state_estimate_error_1 = [] #property of kalman_filter class
mean_state_estimate_2 = []  #property of kalman_filter class
mean_state_estimate_error_2 = [] #property of kalman_filter class

true_mean_state_over_total_time_steps_vehicle_center = []
true_mean_state_over_total_time_steps_gps1 = []
true_mean_state_over_total_time_steps_gps2 = []
true_mean_state_over_total_time_steps_gps3 = []
mean_state_after_prediction_ekf1 = []
mean_state_after_prediction_ekf2 = []
true_mean_state_over_total_time_steps_smartphone= []

mean_state_after_prediction_ekf1.append(ekf1.mean)
mean_state_after_prediction_ekf2.append(ekf2.mean)

for step in range(total_time_steps):
          
    mean_state_estimate_error_1.append(ekf1.cov)
    mean_state_estimate_1.append(ekf1.mean)
    mean_state_estimate_error_2.append(ekf2.cov)
    mean_state_estimate_2.append(ekf2.mean)

    # True mean state at time t = step
    true_mean_state = np.array([[df_piksi_smartphone.loc[step,'lat_piksi_utm_easting']],
                                [df_piksi_smartphone.loc[step,'lon_piksi_utm_northing']]])

    ekf1.predict(dt=time_step)
    ekf2.predict(dt=time_step)
    mean_state_after_prediction_ekf1.append(ekf1.mean)
    mean_state_after_prediction_ekf2.append(ekf2.mean)
    
    if step != 0 and step % measurement_rate == 0:
        #print(step)
        ekf1.update(meas_value=([[df_piksi_smartphone.loc[step,'lat_smartphone_utm_easting']],
                                [df_piksi_smartphone.loc[step,'lon_smartphone_utm_northing']]]),
                                flag = 0)
        if step<len(df_piksi_arduino.loc[:,'lat_GPS1_utm_easting']) and step != 49:
            ekf2.update(meas_value=([[df_piksi_arduino.loc[step,'lat_GPS1_utm_easting']-0],
                                [df_piksi_arduino.loc[step,'lon_GPS1_utm_northing']-1]]),
                                flag = 1)
        if step<len(df_piksi_arduino.loc[:,'lat_GPS2_utm_easting']):
            ekf2.update(meas_value=([[df_piksi_arduino.loc[step,'lat_GPS2_utm_easting']-0.8],
                                [df_piksi_arduino.loc[step,'lon_GPS2_utm_northing']+0.8]]),
                                flag = 2)
        if step<len(df_piksi_arduino.loc[:,'lat_GPS3_utm_easting']):
            ekf2.update(meas_value=([[df_piksi_arduino.loc[step,'lat_GPS3_utm_easting']+0.8],
                                [df_piksi_arduino.loc[step,'lon_GPS3_utm_northing']+0.8]]),
                                flag = 3)

# Create a figure with two subplots
fig, ((ax1, ax3)) = plt.subplots(1, 2, figsize=(10, 5))

# plt.subplot(2, 2, 1)
ax1.set_title('Position')
ax1.set_xlabel('X Position (in m)')
ax1.set_ylabel('Y Position (in m)')
ax1.plot([ms[0] for ms in mean_state_estimate_1], [ms[1] for ms in mean_state_estimate_1], color='indigo', linestyle='--', label = 'Smartphone Pos estimate')
ax1.plot([ms[0] for ms in mean_state_estimate_2], [ms[1] for ms in mean_state_estimate_2],color='salmon', linestyle='--', label = 'Arduino Pos Estimate')
ax1.plot(df_piksi_smartphone.loc[:,'lat_piksi_utm_easting'], df_piksi_smartphone.loc[:,'lon_piksi_utm_northing'], 'k--', label = 'Piksi Pos')
ax1.ticklabel_format(style='plain', useOffset=False)
ax1.tick_params(axis='x', rotation=45)
ax1.legend(loc = 'upper right')

ax3.set_title('Heading')
ax3.set_xlabel('Time (in s)')
ax3.set_ylabel('Heading (in radians)')
ax3.plot([ms[2] for ms in mean_state_estimate_1], color='indigo', linestyle='--', label = 'Smartphone Heading Estimate')
ax3.plot([ms[2] for ms in mean_state_estimate_2], color='salmon', linestyle='--', label = 'Arduino Heading Estimate')
ax3.legend(loc = 'upper right')


fig.tight_layout()
plt.show()



# Calculating the distace from X and Y cordinates
pos_piksi_after_filter_s = np.sqrt(
    np.square(np.array(df_piksi_smartphone.loc[:,'lat_piksi_utm_easting']))
    +np.square(np.array(df_piksi_smartphone.loc[:,'lon_piksi_utm_northing'])))

pos_smartphone_after_filter = np.sqrt(
    np.square(np.array([ms[0] for ms in mean_state_estimate_1]))
    +np.square(np.array([ms[1] for ms in mean_state_estimate_1])))


pos_piksi_after_filter_a = np.sqrt(
    np.square(np.array(df_piksi_arduino.loc[:,'lat_piksi_utm_easting']))
    +np.square(np.array(df_piksi_arduino.loc[:,'lon_piksi_utm_northing'])))


pos_3gps_after_filter = np.sqrt(
    np.square(np.array([ms[0] for ms in mean_state_estimate_2]))
    +np.square(np.array([ms[1] for ms in mean_state_estimate_2]))) 

n = 15
# RMSE for smartphone
mse_smartphone_after_filter = np.square(np.subtract(pos_piksi_after_filter_s[n:],pos_smartphone_after_filter[n:])).mean()
rmse_smartphone_after_filter = np.sqrt(mse_smartphone_after_filter)
# RMSE for arduino    
mse_est_3gps_after_filter = np.square(np.subtract(pos_piksi_after_filter_a[n:], pos_3gps_after_filter[n:])).mean()
rmse_est_3gps_after_filter = np.sqrt(mse_est_3gps_after_filter)


fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))

# Create a list of sensor names and RMSE values
labels = ['Smartphone','Arduino']
values = [rmse_smartphone_after_filter, rmse_est_3gps_after_filter]

# Plot the RMSE values
ax1.bar(labels, values, capsize=5, color='tab:blue', label = 'Dataset-1')
ax1.set_title('RMSE Error Kalman Filter')
ax1.set_xlabel('Sensor Set', fontsize=14)
ax1.set_ylabel('RMSE (in meters)', fontsize=14)
ax1.legend(loc = 'lower right')
for i in range(len(labels)):
    ax1.text(i, values[i]-1, "{:.4f}".format(values[i]), ha='center', va='top')
    

fig.tight_layout()
plt.show()

df_gps1_lat = df_piksi_arduino.loc[:,'lat_piksi_utm_easting']-  df_piksi_arduino.loc[:,'lat_GPS1_utm_easting']
df_gps1_lon = df_piksi_arduino.loc[:,'lon_piksi_utm_northing']-  df_piksi_arduino.loc[:,'lon_GPS1_utm_northing']

df_gps2_lat = df_piksi_arduino.loc[:,'lat_piksi_utm_easting']-  df_piksi_arduino.loc[:,'lat_GPS2_utm_easting']
df_gps2_lon = df_piksi_arduino.loc[:,'lon_piksi_utm_northing']-  df_piksi_arduino.loc[:,'lon_GPS2_utm_northing']

df_gps3_lat = df_piksi_arduino.loc[:,'lat_piksi_utm_easting']-  df_piksi_arduino.loc[:,'lat_GPS3_utm_easting']
df_gps3_lon = df_piksi_arduino.loc[:,'lon_piksi_utm_northing']-  df_piksi_arduino.loc[:,'lon_GPS3_utm_northing']

df_smp_lat = df_piksi_smartphone.loc[:,'lat_piksi_utm_easting']-  df_piksi_smartphone.loc[:,'lat_smartphone_utm_easting']
df_smp_lon = df_piksi_smartphone.loc[:,'lon_piksi_utm_northing']-  df_piksi_smartphone.loc[:,'lon_smartphone_utm_northing']