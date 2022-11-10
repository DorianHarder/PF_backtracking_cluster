import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.random import normal as randnorm
from shapely.geometry import *


def initialize_particles(firstx, firsty, std, particle_number, h_noise, s_noise):
    """
    creating initial particles

    Parameters
    ----------
    firstx : start position x coords - float
    firsty : start position y coords - float
    std : standard deviation for the particle distribution- float
    h_noise : heading noise value - float
    s_noise : step length noise value - float
    particle_number : maximum number of particles - int

    Returns
    -------
    initial_particles : initial particles - geopandas geodataframe
    """
    geoms = []
    length_noise = []  # error for the steplength
    angle_noise = []
    # adding the paticles
    for i in range(particle_number):
        geoms.append(Point(firstx + randnorm(scale=std), firsty + randnorm(scale=std)))
        length_noise.append(randnorm() * s_noise)  # error for the steplength
        angle_noise.append(randnorm() * h_noise)  # heading error

    new_particles_df = pd.DataFrame(
        {'length_noise': length_noise,
         'angle_noise': angle_noise,
         'geometry': geoms})
    new_particles_gdf = gpd.GeoDataFrame(new_particles_df, geometry=new_particles_df.geometry)
    new_particles_gdf.crs = "EPSG:32632"

    return new_particles_gdf
