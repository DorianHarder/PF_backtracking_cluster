import csv
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from Backtracking import BackTracking_LOS
from Particles import initialize_particles  # , get_gdf, initialize
from Step import Step
from data_log import weighted_avg_and_std_geopoints
from input_data import Load_Step_Data, walls, base_plot
# from input_data_win import Load_Step_Data, routing_network, walls, doors, global_rooms, transitions, base_plot
from map_matching import check_floor, CheckParticleSteps_LOS
from matplotlib.artist import Artist
from shapely.geometry import *

# gpd.options.use_pygeos = True



import os
import sys
import pandas as pd
import numpy as np
#from descartes import PolygonPatch
import matplotlib.pyplot as plt
from itertools import combinations
# import alphashape

def find_nearest(array, value):
  array = np.asarray(array)
  idx = (np.abs(array - value)).argmin()
  return idx, array[idx]

def diff_PDR(pdr_position, position):

    # print('pdr_position: ',pdr_position)
    # print('position: ', position)
    diff_pdr = pdr_position.distance(position)
    #print('diff_pdr: ', diff_pdr)

    return diff_pdr

def particle_Precision(points, position):
    #print('points: ', points)
    particle_precision = points


    #points_gdf = gpd.GeoDataFrame(geometry=points)
    #points_gdf.crs = "EPSG:32632"
    #for p in points:
    # points_within = []
    distances = []
    # print('combinations: ', combinations(range(0, points.geometry.shape[0] - 1), 2))
    '''
    for i1, i2 in combinations(range(0, points.geometry.shape[0] - 1), 2):  # Iterate over all pairs/combinations of indices
        print('temp_dist: ',points.geometry[i1].distance(points.geometry[i2]) )
        # points_within.append([points[i1], points[i2]])
        distances.append(points.geometry[i1].distance(points.geometry[i2]))
    '''

    for p in points:  # Iterate over all pairs/combinations of indices
        #print('p1: ', p1)
        #print('p2: ', p2)
        #print('temp_dist: ', p1.distance(p2))
        # points_within.append([points[i1], points[i2]])

        distances.append(position.distance(p))


        #print('distances: ', distances)
        #if len(distances) >= 1:
            #print('distances :', max(distances))
    return max(distances)

def calculate_error2(pos_5g, qualisys, step_time):
    errors = []
    # [[np.nan,np.nan] if np.isnan(p[1]) or any(np.isnan(qualisys[i])) else [abs(p[1] - qualisys[i][1]),abs(p[2] - qualisys[i][2])] for i,p in enumerate(pos_5g)]
    for i, p in enumerate(pos_5g):
        if i < len(step_time):
            idx, ref_time = find_nearest(qualisys[:, 0], step_time[i])
            if abs(ref_time - step_time[i]) < 500:
                if np.isnan(p[0]):
                    errors.append([np.nan])
                elif np.isnan(qualisys[idx][1]):
                    errors.append([np.nan])
                else:
                    x_error = abs(p[0] - qualisys[idx][1])
                    y_error = abs(p[1] - qualisys[idx][2])
                    error = np.sqrt(x_error ** 2 + y_error ** 2)
                    # z_error = abs(p[3] - qualisys[i][3])
                    errors.append([error])

    return errors



    '''
    for p in points.geometry:


        distances.append(position.distance(p))
    return max(distances)
    '''






def determine_uncertainty(pdr_position, points, position,particles):
    diff_pdr = diff_PDR(pdr_position, position)


    if len(points) > 1:
        particle_precision = particle_Precision(points, position)

    else:
        particle_precision = particle_Precision(particles.geometry, position)


    return diff_pdr, particle_precision