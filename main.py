import csv
import time
import matplotlib.lines as mlines
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from Backtracking import BackTracking_LOS, BackTracking_rooms, BackTracking_routing
from Particles import initialize_particles  # , get_gdf, initialize
from Step import Step
from data_log import weighted_avg_and_std_geopoints, write_logfile, write_particles_to_csv
from input_data import Load_Step_Data, walls, base_plot
# from input_data_win import Load_Step_Data, routing_network, walls, doors, global_rooms, transitions, base_plot
from map_matching import check_floor, CheckParticleSteps_LOS, floor_update
from matplotlib.artist import Artist
from shapely.geometry import *
from uncertainty import determine_uncertainty,calculate_error2
from Propagation import propagate_PDR  # '(temp_step)
#import warnings
#from shapely.errors import ShapelyDeprecationWarning

#warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# gpd.options.use_pygeos = True

from clustering import cluster_mean_shift
from mathematics import calc_azimuth

import os
import sys
import pandas as pd
# import numpy as np
# from descartes import PolygonPatch
# import matplotlib.pyplot as plt
# import alphashape

from plotting import base_figure_plot, update_plot, final_plot



sys.path.insert(0, os.path.dirname(os.getcwd()))

def updt(total, progress,name):
    """
    Displays or updates a console progress bar.

    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
        progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r " + name + ": [{}] {:.0f}% {}".format(
    #text = "\r syncing 5G time stamps: [{}] {:.0f}% {}".format(
        "#" * block + "-" * (barLength - block), round(progress * 100, 0),
        status)
    sys.stdout.write(text)
    sys.stdout.flush()


def run(_trajectory, concept, s_scale,rate, routing_support=False):
    """
    mapmatching and backtracking process for each step. Resampling and weighting by routing can be uncommented
    or left inactive. The live plotting could also be deactivated by commenting it out to increase performance slightly

    Parameters
    ----------
    _trajectory : choose the trajectory - string
    concept : choose concept of particle checking - string
    s_scale : step length correction - float
    routing_support : activate or deactivate weighting by routing - Boolean Value, default is False


    Returns
    -------
    positions_list : x and y coordinates and height value for each step - list of lists (of floats)
    std_list : the positions standard deviation for each step - list of floats
    trajectory : list of the positions for each step - list of shapely point geometries
    """

    list_of_positions = []
    list_of_stds = []
    positions_of_trajectory = []
    pdr_trajectory = []
    userheight = 1.7
    #particle_list = []

    # parameter dependent on the given trajectory:
    if _trajectory == 'wls':
        # eight-path:
        s_heading, s_length, ref_eight, s_height, gs_ref_eight, kf_points_gdf,g5_list,step_time = Load_Step_Data('kf',rate)
        g5_points = [Point([g5[1],g5[2]]) for g5 in g5_list]
        gs_g5 = gpd.GeoSeries(g5_points)
        heading0 = s_heading[0]
        last_heading_estimate = heading0
        start_height = 0 + .8 * userheight
        current_floor = 0
        step_scale = s_scale
        base = walls[0]
        x_0 = ref_eight[0][1]
        y_0 = ref_eight[0][2]
        start = Point(x_0, y_0)  # startposition
        calculated_position = start
        pdr_trajectory.append(start)
        x_ref = [g[1] for g in ref_eight if np.isnan(g[1]) == False]
        y_ref = [g[2] for g in ref_eight if np.isnan(g[1]) == False]
        ref_line = gpd.GeoSeries(LineString([[x_ref[i],y_ref[i]] for i in range(len(x_ref))]))
        
        x_kf = [g.coords[0][0] for g in kf_points_gdf.geometry if np.isnan(g.coords[0][0]) == False]
        y_kf = [g.coords[0][1] for g in kf_points_gdf.geometry if np.isnan(g.coords[0][0]) == False]
        kf_line = gpd.GeoSeries(LineString([[x_kf[i], y_kf[i]] for i in range(len(x_kf))]))


        calculated_position_gdf = gpd.GeoDataFrame(geometry=[calculated_position])
        calculated_position_gdf.crs = "EPSG:32632"
        list_of_positions.append((calculated_position.coords[0][0], calculated_position.coords[0][1], start_height))


    current_height = start_height
    max_num_particles = 200
    std = 1  # standard deviation for initial particle position

    r_rout = 1.5 ** 2  # noise for routing support
    h_noise = 5 * np.pi / 180  # heading noise, not used for the current propagation
    s_noise = 0.05  # step length noise, not used for the current propagation

    # initial Particles (befor first step):

    start_particles = initialize_particles(x_0, y_0, std, max_num_particles, h_noise, s_noise)

    # all particles each step
    particles = start_particles
    steps = []

    floor_data = walls[0]

    # plot:
    fig, ax, current_plot, frame = base_figure_plot(base)

    # last_transition = transitions[0]
    everything_start_time = time.time()

    for s in range(len(s_heading)):
        last_position = calculated_position
        #print(s)

        # calculating height and checking for current floor and floor change:

        current_height, floor_changed, last_floor, current_floor, current_floor_data = floor_update(
                s_height[s], current_height, current_floor, floor_data)

        if current_floor != last_floor:
            floor_changed = True
            #current_transition = gpd.overlay(current_transition, last_transition, how='intersection')

            # only for rooms concept:

        # creating step object
        temp_step = Step(s_length[s], s_heading[s], current_height, current_floor, calculated_position, _scale=step_scale)

        pdr_position, dx_pdr, dy_pdr = propagate_PDR(pdr_trajectory[s], temp_step)
        pdr_trajectory.append(pdr_position)
        heading_change_pdr = s_heading[s] - s_heading[s - 1]

        particles = CheckParticleSteps_LOS(particles, temp_step, current_floor_data)


        steps.append(temp_step)
        particles.reset_index(drop=True)  # resetting index of the particle dataframe

        # clustering of particles:

        coords = [[p.coords[0][0], p.coords[0][1]] for p in particles.geometry]
        #particle_list.append(coords)
        if len(coords) > 1:
            winner_cluster, winner_centroid, separated_clusters, cluster_centroids, cluster_heading = cluster_mean_shift(
                list_of_positions[s], coords, heading_change_pdr, last_heading_estimate, dx_pdr, dy_pdr, floor_changed,
                correction=False)

        # weighting:
        particle_number = len(particles.geometry)
        w0 = np.full(particle_number, 1 / particle_number)


        w = w0
        w = np.divide(w, np.sum(w))
        # standarddeviation,logging and calculation of estimated position:
        calculated_position, x_std, y_std = weighted_avg_and_std_geopoints(gpd.GeoDataFrame(geometry=winner_cluster), w)

        diff_pdr, particle_precision = determine_uncertainty(pdr_position, cluster_centroids, calculated_position,
                                                             particles)

        positions_of_trajectory.append(calculated_position)  # calculated position for this step

        list_of_positions.append((calculated_position.coords[0][0], calculated_position.coords[0][1], temp_step.height))
        list_of_stds.append((x_std, y_std, 0))

        last_heading_estimate = np.deg2rad(calc_azimuth(last_position.coords[0], (calculated_position.coords[0])))

        # step correction calculator (used when floor is changed):
        # temp_step_scale = np.mean(temp_particles.length_noise)
        # if floor_changed:
        # step_scale += temp_step_scale




        # ###########backtracking:###################################

        # backtracking parameter:

        try_time_max = 8
        back_tracking_step_number = min(32, s) + 1

        new_particle_number = max_num_particles - particle_number

        if back_tracking_step_number == 1:
            back_tracking_steps = [
                steps[-1]]  # neccessary for first step in eight path, because "for b in range(1,1) doesn't work"
        else:
            back_tracking_steps = [steps[-b] for b in range(1, back_tracking_step_number) if
                                   steps[-b].current_floor == temp_step.current_floor]

        # print('back_tracking_len ', len(back_tracking_steps))

        # generating new particles after checking the backtracking of the particles

        if concept == 'LOS':
            particles = BackTracking_LOS(new_particle_number, particles, back_tracking_steps, current_floor_data,
                                         try_time_max,
                                         h_noise, s_noise)

        # #######plot:
        fig, ax, current_plot, frame = update_plot(fig, ax, current_plot, frame, concept, base_plot, current_floor, calculated_position,
                                                   particle_precision,
                                                   separated_clusters, winner_cluster,
                                                   diff_pdr, s_heading, heading0, s, std, _trajectory,ref_line,kf_line,kf_points_gdf,gs_g5,rate)
        updt(len(s_heading), s + 1, 'steps')
    final_plot(base_plot, positions_of_trajectory, _trajectory, s_scale,ref_line, kf_line, kf_points_gdf,list_of_positions,gs_g5,rate)


    print("---Whole Tracjectory %s seconds ---" % (time.time() - everything_start_time))

    errors_1 = calculate_error2(np.array(list_of_positions), np.array(ref_eight)[:, 0:3], step_time)
    data_1 = np.c_[step_time, np.array(list_of_positions), errors_1]
    err = [e for e in errors_1 if np.isnan(e) == False]
    err.sort()
    sorted_err = np.array(err)
    err_range = [i / len(np.arange(sorted_err.size)) for i in np.arange(sorted_err.size)]

    
    
    
    
    
    # writing into file:
    #filename_run = 'output/wls/BT_wls_{}_{}.csv'.format( concept, str(rate))
    np.savetxt('output/wls/BT_wls_{}_{}.csv'.format( concept, str(rate)), data_1, delimiter=',', header='time,x,y,z,error')
    fontsize = 32
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(sorted_err, err_range, color='red', label="CDF")
    ax.set_title("CDF of Positionestimates from Map Matching", fontsize=fontsize)
    ax.set_xlabel("error [m]", weight='normal',
                  size=fontsize,
                  labelpad=6)
    ax.set_ylabel("probability ", weight='normal',
                  size=fontsize,
                  labelpad=6)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.legend(fontsize=fontsize)
    fig.savefig('output/wls/BT_wls_{}_{}_CDF.png'.format( concept, str(rate)), dpi=fig.dpi)
    # plt.show()
    
    #write_logfile(filename_run, list_of_positions, list_of_stds)

    #filename_particles = 'particles.csv'
    #write_particles_to_csv(filename_particles, particle_list)

    return list_of_positions, list_of_stds, positions_of_trajectory,step_time


# ################# run the PF process: ##############
# positions_list, std_list, trajectory = run('eight','LOS',0,routing_support = False)
positions_list, std_list, trajectory,step_time = run('wls', 'LOS', 0,2, routing_support=False)


