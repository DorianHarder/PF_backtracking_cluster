#from scipy.cluster.vq import kmeans2, whiten
#from sklearn.cluster import DBSCAN  # clustering
from shapely.geometry import *
import geopandas as gpd
import numpy as np
from mathematics import azimuthAngle, calc_azimuth
from sklearn.cluster import MeanShift, estimate_bandwidth


def cluster_mean_shift(last_position, coords, heading_change_PDR, last_heading_estimate, dx_PDR, dy_PDR,floor_changed,correction):
    bandwidth = estimate_bandwidth(coords, quantile=0.3)
    #print('coords: ',coords)
    if bandwidth == 0:
        bandwidth = 0.01
    #print('bandwidth: ', bandwidth)
    #bandwidth = #3#1#2#5'
    clustering = MeanShift(bandwidth=bandwidth).fit(coords)
    cluster_labels = clustering.labels_
    number_of_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    #params = clustering.get_params(deep=True)
    #print('params: ', params)

    clusters = []
    cluster_centers = []
    for c in range(number_of_clusters):
        clusters.append([Point(coords[i]) for i, l in enumerate(cluster_labels) if l == c])
        cluster_coords_x = np.mean([coords[i][0] for i, l in enumerate(cluster_labels) if l == c])
        cluster_coords_y = np.mean([coords[i][1] for i, l in enumerate(cluster_labels) if l == c])
        cluster_centers.append(Point(cluster_coords_x, cluster_coords_y))

    
    master_centroid = Point(last_position[0], last_position[1])
    master_cluster = [Point(last_position[0], last_position[1])]
    clusters.sort(key=len)
    # master_centroid = MultiPoint(master_cluster).centroid
    #print('clusters 1st - second: ', len(clusters[0]),len(clusters[-1]))
    #cluster_headings = []
    count = 0
    if correction == True:
        #if floor_changed == False:
        for cc in clusters:

            if len(cc) > 0:
                #cc_centroid = MultiPoint(cc).centroid
                cluster_heading = calc_azimuth((last_position[0], last_position[1]), MultiPoint(cc).centroid.coords[0])
                #cluster_headings.append(cluster_heading)
                cluster_heading_change_rate = cluster_heading - last_heading_estimate
                #temp_x = last_position[0] + dx_PDR
                #temp_y = last_position[1] + dy_PDR


                if abs(heading_change_PDR - cluster_heading_change_rate) <= np.deg2rad(90):


                #if abs(temp_x - cc_centroid.coords[0][0] )< 5 and abs(temp_y - cc_centroid.coords[0][1] )< 5:

                    #print('min one valid cluster')

                    master_cluster = cc
                    master_centroid = MultiPoint(cc).centroid
                    count +=1
        #else:
            #for cc in clusters:
                #master_cluster = cc
                #master_centroid = MultiPoint(cc).centroid


    else:

        master_cluster = max(clusters, key=len)
        master_centroid = MultiPoint(master_cluster).centroid
        cluster_heading = calc_azimuth((last_position[0], last_position[1]), master_centroid.coords[0])

    master_cluster_points = [Point(coo) for coo in master_cluster]




    return master_cluster_points, master_centroid, clusters, cluster_centers,cluster_heading#,cluster_headings


