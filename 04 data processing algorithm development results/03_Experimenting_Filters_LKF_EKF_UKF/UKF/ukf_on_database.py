# -*- coding: utf-8 -*-
"""
Created on Mon May 29 11:18:09 2023

@author: Z0168020
"""

from UKF_4_state_3_meas_OneByOneUpdate import UKF

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians, acos
import pyproj

# function to calculate the distance between two gps co-ordinates
def dist_in_meters(lat1, lon1, lat2, lon2):
    R = 6371000  # radius of the Earth in meters
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad
    
    # a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
    # c = 2 ⋅ atan2( √a, √(1−a) )
    
    a = (sin(delta_lat / 2) ** 2 +
         cos(lat1_rad) * cos(lat2_rad) *
         sin(delta_lon / 2) ** 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

# function to convert gps coordinates into UTM coordinates
def convert_to_utm(df, lat_col, long_col):
    utm_proj = pyproj.Proj(proj='utm', zone=32, ellps='WGS84') # set UTM projection with zone number and ellipsoid
    utm_coords = []
    for lat, lon in zip(df[lat_col], df[long_col]):
        easting, northing = utm_proj(lon, lat) # convert latitude and longitude to UTM easting and northing
        utm_coords.append((easting, northing))
    df[lat_col + '_utm_easting'], df[long_col+'_utm_northing'] = zip(*utm_coords) # add new columns to the DataFrame
    return df

# function to convert UTM coordinates into GPS coordinates
def convert_to_gps(df, easting_col, northing_col):
    utm_proj = pyproj.Proj(proj='utm', zone=32, ellps='WGS84') # set UTM projection with zone number and ellipsoid
    gps_proj = pyproj.Proj(proj='latlong', datum='WGS84') # set GPS projection with WGS84 datum
    gps_coords = []
    for easting, northing in zip(df[easting_col], df[northing_col]):
        lon, lat = pyproj.transform(utm_proj, gps_proj, easting, northing) # convert UTM easting and northing to latitude and longitude
        gps_coords.append((lat, lon))
    df['Latitude'], df['Longitude'] = zip(*gps_coords) # add new columns to the DataFrame
    return df

def angle_between_coordinates(x1, y1, x2, y2, x3, y3, x4, y4):
    dx1 = x2 - x1
    dy1 = y2 - y1
    dx2 = x4 - x3
    dy2 = y4 - y3
    dot_product = dx1 * dx2 + dy1 * dy2
    magnitude1 = sqrt(dx1**2 + dy1**2)
    magnitude2 = sqrt(dx2**2 + dy2**2)
    cos_angle = dot_product / (magnitude1 * magnitude2)
    angle = np.degrees(acos(cos_angle))
    return angle

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
num_states = 4
measu_inputs = 2
control_inputs = 2
rad_wheel =  0.055

init_state_ard = np.array([681841, 5406351, 1.2, 0.005]).reshape(num_states,1)
init_state_smp = np.array([681844, 5406341, 1.1, 0.002]).reshape(num_states,1)

init_cov = np.eye(num_states)
u = np.array([36.364, 
              0.005]).reshape(2,1)

process_variance = np.array([[1.5,  0,  0,     0],
                             [0,  1.5,  0,     0],
                             [0,    0,  0.0015, 0],
                             [0,    0,    0,     0.000015]])
measu_variance_ard = 0.01
measu_variance_smp = 0.02

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


mean_state_estimate_1 = []  #property of kalman_filter class
mean_state_estimate_error_1 = [] #property of kalman_filter class
mean_state_estimate_2 = []  #property of kalman_filter class
mean_state_estimate_error_2 = [] #property of kalman_filter class

true_mean_state_over_total_time_steps_vehicle_center = []
true_mean_state_over_total_time_steps_gps1 = []
true_mean_state_over_total_time_steps_gps2 = []
true_mean_state_over_total_time_steps_gps3 = []
mean_state_after_prediction = []
true_mean_state_over_total_time_steps_smartphone= []


for step in range(total_time_steps):
          
    mean_state_estimate_error_1.append(ukf1.cov)
    mean_state_estimate_1.append(ukf1.mean)
    mean_state_estimate_error_2.append(ukf2.cov)
    mean_state_estimate_2.append(ukf2.mean)

    # True mean state at time t = step
    true_mean_state = np.array([[df_piksi_smartphone.loc[step,'lat_piksi_utm_easting']],
                                [df_piksi_smartphone.loc[step,'lon_piksi_utm_northing']]])

    ukf1.predict(dt=time_step)
    ukf2.predict(dt=time_step)
    mean_state_after_prediction.append(ukf1.mean)
    
    if step != 0 and step % measurement_rate == 0:
        #print(step)
        ukf1.update(z=([[df_piksi_smartphone.loc[step,'lat_smartphone_utm_easting']],
                                [df_piksi_smartphone.loc[step,'lon_smartphone_utm_northing']]]),
                                dt = time_step,
                                flag = 0)
        if step<len(df_piksi_arduino.loc[:,'lat_GPS1_utm_easting']) and step != 49:
            ukf2.update(z=([[df_piksi_arduino.loc[step,'lat_GPS1_utm_easting']-0],
                                [df_piksi_arduino.loc[step,'lon_GPS1_utm_northing']-1]]),
                                dt = time_step,
                                flag = 1)
        if step<len(df_piksi_arduino.loc[:,'lat_GPS2_utm_easting']):
            ukf2.update(z=([[df_piksi_arduino.loc[step,'lat_GPS2_utm_easting']-0.8],
                                [df_piksi_arduino.loc[step,'lon_GPS2_utm_northing']+0.8]]),
                                dt = time_step,
                                flag = 2)
        if step<len(df_piksi_arduino.loc[:,'lat_GPS3_utm_easting']):
            ukf2.update(z=([[df_piksi_arduino.loc[step,'lat_GPS3_utm_easting']+0.8],
                                [df_piksi_arduino.loc[step,'lon_GPS3_utm_northing']+0.8]]),
                                dt = time_step,
                                flag = 3)

# Create a figure with two subplots
fig, ((ax1, ax3)) = plt.subplots(1, 2, figsize=(10, 5))

# plt.subplot(2, 2, 1)
ax1.set_title('Position')
ax1.set_xlabel('Time (in s)')
ax1.set_ylabel('X-Y Position (in m)')
ax1.plot([ms[0] for ms in mean_state_estimate_1], [ms[1] for ms in mean_state_estimate_1], color='indigo', marker= 'o', linestyle='--', label = 'Smartphone Pos estimate')
ax1.plot([ms[0] for ms in mean_state_estimate_2], [ms[1] for ms in mean_state_estimate_2], color='salmon', marker= 'o', linestyle='--', label = 'Arduino Pos Estimate')
ax1.plot(df_piksi_smartphone.loc[:,'lat_piksi_utm_easting'], df_piksi_smartphone.loc[:,'lon_piksi_utm_northing'], 'k--', label = 'Piksi Pos')
ax1.ticklabel_format(style='plain', useOffset=False)
ax1.tick_params(axis='x', rotation=45)
ax1.legend(loc = 'upper right')

ax3.set_title('Heading')
ax3.set_xlabel('Time (in s)')
ax3.set_ylabel('Heading (in radians)')
ax3.plot([ms[2] for ms in mean_state_estimate_1],  color='indigo', linestyle='--', label = 'Smartphone Heading Estimate')
ax3.plot([ms[2] for ms in mean_state_estimate_2],  color='salmon', linestyle='--', label = 'Arduino Heading Estimate')
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

# RMSE for smartphone
mse_smartphone_after_filter = np.square(np.subtract(pos_piksi_after_filter_s[:],pos_smartphone_after_filter[:])).mean()
rmse_smartphone_after_filter = np.sqrt(mse_smartphone_after_filter)
# RMSE for arduino    
mse_est_3gps_after_filter = np.square(np.subtract(pos_piksi_after_filter_a[:], pos_3gps_after_filter[:])).mean()
rmse_est_3gps_after_filter = np.sqrt(mse_est_3gps_after_filter)


fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))

# Create a list of sensor names and RMSE values
labels = ['Smartphone','3 GPS Fusion']
values = [rmse_smartphone_after_filter, rmse_est_3gps_after_filter]

# Plot the RMSE values
ax1.bar(labels, values, capsize=5, color='tab:blue', label = 'Dataset-1')
ax1.set_title('RMSE Error Kalman Filter')
ax1.set_xlabel('Sensor Set', fontsize=14)
ax1.set_ylabel('RMSE (in meters)', fontsize=14)
ax1.legend(loc = 'lower right')
for i in range(len(labels)):
    ax1.text(i, values[i]+1, "{:.4f}".format(values[i]), ha='center', va='top')

fig.tight_layout()
plt.show()