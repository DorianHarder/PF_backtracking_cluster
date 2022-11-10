import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import *


def Propagation(_particles, _sheading, _slength):
    """
    propagation of all particles

    Parameters
    ----------
    _particles : particles (including position, scale and angle) - geopandas geodataframe
    _sheading : step heading - float
    _slength : step length or not - float

    Returns
    -------
    propagated_particles : valid propagated particles - geopandas geodataframe
    """
    SH = _sheading
    SL = _slength
    particles = _particles
    # in principle like the propagation from thomas, but with the same error (length_noise) for x and y direction.
    propagated_geom = [Point(
        p.coords[0][0] + np.cos(SH + particles.angle_noise[i]) * (SL + particles.length_noise[i]),
        p.coords[0][1] + np.sin(SH + particles.angle_noise[i]) * (SL + particles.length_noise[i])) for
        i, p in enumerate(particles.geometry)]
    propagated_length_noise = [s for s in particles.length_noise]
    propagated_angle = [a for a in particles.angle_noise]

    propagated_particles_df = pd.DataFrame(
        {'length_noise': propagated_length_noise,
         'angle_noise': propagated_angle,
         'geometry': propagated_geom})
    propagated_particles = gpd.GeoDataFrame(propagated_particles_df, geometry=propagated_particles_df.geometry)
    propagated_particles.crs = "EPSG:32632"

    return propagated_particles


def propagate_PDR(last_position, _step):
    SH = _step.heading
    SL = _step.length
    dx = np.cos(SH) * (SL)
    dy = np.sin(SH) * (SL)
    propagated_geom = Point(
        last_position.coords[0][0] + dx,
        last_position.coords[0][1] + dy)
    '''
    propagated_PDR_position = pd.DataFrame(
        {'geometry': propagated_geom})
    propagated_PDR_position = gpd.GeoDataFrame(propagated_PDR_position, geometry=propagated_PDR_position.geometry)
    propagated_PDR_position.crs = "EPSG:32632"
    '''
    return propagated_geom, dx, dy# propagated_PDR_position