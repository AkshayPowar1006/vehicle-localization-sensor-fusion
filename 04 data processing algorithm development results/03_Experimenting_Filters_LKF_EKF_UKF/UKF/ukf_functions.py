# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:20:16 2023

@author: Z0168020
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians, acos
import pyproj


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
    df['deltaT'] = cum_diff_time_s
    df['deltaT'] = df['deltaT'].fillna(1)
    
    
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