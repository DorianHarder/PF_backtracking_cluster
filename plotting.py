import csv
import time
import matplotlib.lines as mlines
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from Backtracking import BackTracking_LOS
from Particles import initialize_particles  # , get_gdf, initialize
from Step import Step
from data_log import weighted_avg_and_std_geopoints
from input_data import Load_Step_Data, walls, base_plot

from map_matching import check_floor, CheckParticleSteps_LOS
from matplotlib.artist import Artist
from shapely.geometry import *
from uncertainty import determine_uncertainty
from Propagation import propagate_PDR#'(temp_step)



from clustering import cluster_mean_shift
from mathematics import calc_azimuth

import os
import sys
import pandas as pd
import numpy as np
#from descartes import PolygonPatch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.getcwd()))

def base_figure_plot(base):
    fig = plt.figure(figsize=(20, 20))
    ax = plt.subplot()
    plt.ion()
    base.plot(ax=ax, color='black')

    frame = fig.text(0.2, 0.8, "", size=9,
                     ha="left", va="top",
                     )
    fig.canvas.draw()
    fig.show()
    plt.pause(0.5)
    return fig, ax,  plt, frame
    
def update_plot(fig, ax,  plt, frame, concept, base_plot, current_floor, calculated_position, particle_precision, separated_clusters, winner_cluster, diff_pdr,Heading,heading0, s, std, _trajectory, ref,kf,kf_points,gs_g5,rate):
    ax.clear()
    fontsize=32
    if current_floor == 0:
        base_plot[0].plot(ax=ax, facecolor="lightgrey",linewidth=1, edgecolor='black')

    elif current_floor == 1:
        base_plot[0].plot(ax=ax, color='black')


    elif current_floor == 4:
        base_plot[0].plot(ax=ax, color='black')


    ref.plot(ax=ax,color='limegreen', label='Ground Truth', zorder=0,linewidth=4)

    #kf_points.plot(ax=ax, marker='o', markersize=50,facecolor="royalblue",linewidth=1, label='WLS Optimisation',zorder=2, alpha=0.5)
    ## kf.plot(ax=ax,linestyle=':', color = 'royalblue',linewidth=6, label='WLS Optimisation',zorder=1, alpha=0.5)
    ## gs_g5.plot(ax=ax, color='grey', marker="o", markersize=50, label='5G Coordinates', zorder=1,edgecolor='black')

    if len(separated_clusters) > 0:
        for cl in separated_clusters:
            gpd.GeoSeries(cl).plot(ax=ax, color='lightgrey', edgecolor='darkgrey', markersize=25, zorder=3)
    # particles.plot(ax = ax, color='lightgrey', edgecolor='darkgrey',markersize=2,zorder=3)
    gpd.GeoSeries(winner_cluster).plot(ax=ax, color='green', edgecolor='darkgreen', markersize=25, zorder=3)


    # SHeading
    heading_line_x = calculated_position.coords[0][0] + np.cos(Heading[s]) * 5
    heading_line_y = calculated_position.coords[0][1] + np.sin(Heading[s]) * 5
    gpd.GeoSeries(LineString([calculated_position.coords[0], (heading_line_x, heading_line_y)])).plot(ax=ax,
                                                                                                      color='black',
                                                                                                  zorder=4)

    ax.set_xlim(left=5, right=35)
    ax.set_ylim(bottom=0, top=26)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    #traj_diff = mlines.Line2D([], [], color='blue', linestyle='-',
    #                              linewidth=1.5, label='particle precision')
    reference = mlines.Line2D([], [],  color='limegreen',linewidth=4, label='Reference')
    #optimisation = mlines.Line2D([], [], marker='o', markersize=50,markerfacecolor="royalblue",markeredgewidth=1,linestyle=':', alpha=0.5, color='lightgrey', label='Optimisation')
    ## optimisation = mlines.Line2D([], [], linestyle=':', alpha=0.5, color='royalblue', label='Optimisation',linewidth=6)

    chosen_cluster = mlines.Line2D([], [], color='green', marker='o',
                                   markersize=8, label='chosen cluster', linestyle='None')
    discarded_cluster = mlines.Line2D([], [], color='lightgrey', marker='o',markeredgecolor='darkgrey',
                                      markersize=8, label='discarded cluster', linestyle='None')
    g5kf_points =  mlines.Line2D([], [],color='grey', marker='o', markersize=13,
                        label='5G Positions', linestyle='None',markeredgecolor='black')
    ## plt.legend(handles=[chosen_cluster, discarded_cluster,reference,optimisation,g5kf_points], edgecolor='darkgrey',fontsize=fontsize)
    plt.legend(handles=[chosen_cluster, discarded_cluster, reference], edgecolor='darkgrey',
               fontsize=fontsize, loc='lower left',)

    gpd.GeoSeries(calculated_position).plot(ax=ax, color='black', markersize=200, zorder=4)
    # cluster_centroids.plot(ax=ax, color='red', markersize=5)
    Artist.remove(frame)
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    textstr = '\n'.join((f'Step: {s}',
                         # f'roundnes: {cluster_roundnes[0]:.2f}',
                         f'standard deviation: {std} m',
                         f'diff. to optimised trajectory: {diff_pdr:.2f} m',
                         f'particle precision: {particle_precision:.2f} m',
                         f'number of clusters: {len(separated_clusters)}'
                         ))
    frame = fig.text(0.13, 0.65, textstr, size=fontsize,
                     ha="left", va="top", bbox=props
                     )
    Artist.set_visible(frame, False)
    # To remove the artist

    fig.canvas.draw()
    plt.savefig(f"output/wls/gif_frames/BT_wls_{str(rate)}s_{str(s)}.jpg")

    plt.pause(0.5)
    
    return fig, ax,  plt, frame

def final_plot(base_plot, trajectory, _trajectory,s_scale,ref,kf,kf_points,list_of_positions,gs_g5,rate):
    fig, ax = plt.subplots(figsize=(20, 20))
    base_plot[0].plot(ax=ax, facecolor="lightgrey",linewidth=1, edgecolor='black')
    fontsize = 32
    x = [p[0] for p in list_of_positions]
    y = [p[1] for p in list_of_positions]
    ax.plot(x,y, linestyle='-', color='red',markeredgecolor='black', markersize=10, label='Trajectory', zorder=4)
    gpd.GeoSeries(trajectory).plot(ax=ax, marker='o', color='red',markersize=10, label='Trajectory', zorder=4)

    ref.plot(ax=ax, linestyle='-', color='limegreen', label='Reference', zorder=3)

    kf_points.plot(ax=ax, marker='o', markersize=10, facecolor="royalblue", linewidth=1, edgecolor='black',
                   label='KF Optimisation', zorder=2, alpha=0.5)
    kf.plot(ax=ax, linestyle=':', color='royalblue', label='KF Optimisation', zorder=1, alpha=0.5)
    gs_g5.plot(ax=ax, marker='o', markersize=25, color='k', linewidth=1,
                        label='5G Positions', zorder=2)

    reference = mlines.Line2D([], [], color='limegreen', label='Reference')
    pf_pos_trajectory = mlines.Line2D([], [], marker='o', markersize=3, markerfacecolor="black", markeredgewidth=1,
                                 markeredgecolor='red', linestyle='-', color='red', label='PF Position Estimates')

    optimisation = mlines.Line2D([], [], marker='o', markersize=3, markerfacecolor="royalblue", markeredgewidth=1, linestyle=':', color='royalblue', label='Optimisation',  alpha=0.5)

    g5kf_points = mlines.Line2D([], [], color='k', marker='o', markersize=6,
                                label='5G Positions', linestyle='None', markeredgecolor='black')
    plt.legend(handles=[pf_pos_trajectory,reference, optimisation, g5kf_points], edgecolor='darkgrey',fontsize=fontsize)


    fig.canvas.draw()
    plt.savefig(f"output/wls/PF_BT_wls_{str(rate)}s.jpg")