import math

import pandas as pd
from input_data import *
from numpy import cos as ncos
from numpy import sin as nsin
from numpy.random import normal as randnorm
from numpy.random import uniform as randuniform

gpd.options.use_pygeos = True

def Resampling(w, propagated_particles):
    """
    resampling of the weights: use a new distribution once the weights are too low

    Parameters
    ----------
    w : normalized w - numpy array of float weights
    propagated_particles : particle coordinates for the current step - geopandas type

    Returns
    -------
    resampled or propagated_particles: new particle coordinates - geopandas type
    """

    top = []

    top.append(w[0])
    for i in range(len(propagated_particles.geometry) - 1):
        top.append(top[i] + w[i + 1])

    resampled_geom = []
    resampled_length_noise = []
    resampled_angle = []
    for i in range(len(propagated_particles.geometry)):
        zf = np.random.rand(1)  # draw uniformly distributed random number

        for jj in propagated_particles.index:
            if top[jj] > zf:
                new_x = propagated_particles.geometry[jj].coords[0][0]
                new_y = propagated_particles.geometry[jj].coords[0][1]
                temp_angle, temp_radius = randuniform(-math.pi, math.pi), randuniform(0.1,
                                                                                      0.5)  # random radius around particle for resampling (like in backtracking)

                resampled_geom.append(
                    Point(new_x + ncos(temp_angle) * temp_radius, new_y + nsin(temp_angle) * temp_radius))
                resampled_length_noise.append(propagated_particles.length_noise[jj])
                resampled_angle.append(propagated_particles.angle_noise[jj])
                break

    resampled_particles_gdf = pd.DataFrame(
        {
            'length_noise': resampled_length_noise,
            'angle_noise': resampled_angle,
            'geometry': resampled_geom})
    new_particles_gdf = gpd.GeoDataFrame(resampled_particles_gdf, geometry=resampled_particles_gdf.geometry)
    new_particles_gdf.crs = "EPSG:32632"

    return new_particles_gdf