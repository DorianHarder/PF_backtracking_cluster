import math

import geopandas
import numpy as np
from shapely.geometry import *
import csv


def weighted_avg_and_std_geopoints(points_layer, weights):
    """
    calculation of position estimate and it's std
    Parameters
    ----------
    points_layer: particle positions - GeoDataFrame of Point-geometries
    weights: particle weights - array of floats

    Returns
    -------
    Point(x_average, y_average): calculated position - shapely Point-geometrie
    math.sqrt(x_variance): std of x-coordinates - float
    math.sqrt(y_variance): std of y-coordinates - float
    """
    xvalues = np.array([])
    yvalues = np.array([])

    for i, points in points_layer.iterrows():
        xvalues = np.append(xvalues, points.geometry.coords[0][0])
        yvalues = np.append(yvalues, points.geometry.coords[0][1])
    x_average = np.average(xvalues, weights=weights[0:len(xvalues)])
    y_average = np.average(yvalues, weights=weights[0:len(yvalues)])
    x_variance = np.average((xvalues - x_average) ** 2, weights=weights[0:len(xvalues)])
    y_variance = np.average((yvalues - y_average) ** 2, weights=weights[0:len(yvalues)])

    return Point(x_average, y_average), math.sqrt(x_variance), math.sqrt(y_variance)


def write_logfile(filename, positions_list, std_list):
    with open(filename, mode='w') as run_file:
        run_file_writer = csv.writer(run_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # run_file_writer.writerow(['all particles that are not in transition zones are deleted with floor change'])
        run_file_writer.writerow(['X', 'Y', 'Z', 'X_std', 'Y_std', 'Z_std'])
        print('writing1')
        for pl in range(len(positions_list) - 1):
            run_file_writer.writerow(
                [str(positions_list[pl + 1][0]), str(positions_list[pl + 1][1]), str(positions_list[pl + 1][2]),
                 str(std_list[pl][0]),
                 str(std_list[pl][1]), str(std_list[pl][2])])

    return


def write_particles_to_csv(filename, particle_list):
    with open(filename, mode='w') as particle_file:
        particle_file_writer = csv.writer(particle_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for p in range(len(particle_list)):

            # c_row = []
            for pp in particle_list[p]:
                particle_file_writer.writerow([str(pp[0]), str(pp[1])])

            particle_file_writer.writerow([' ', ' '])

    return
