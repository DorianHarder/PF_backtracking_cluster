import geopandas as gpd
import numpy as np
import pandas as pd
from Propagation import Propagation
from shapely.geometry import *

# gpd.options.use_pygeos = True





def CheckParticleSteps_LOS(particles, step, floor_data):
    # check the propagation of the particles for the current step for validity, by line of sight
    last_x, last_y = [l.coords[0][0] for l in particles.geometry], [l.coords[0][1] for l in particles.geometry]
    propagated_particles = Propagation(particles, step.heading, step.length)

    Lines = [LineString([(last_x[i], last_y[i]), (
    propagated_particles.geometry[i].coords[0][0], propagated_particles.geometry[i].coords[0][1])]) for i in
             propagated_particles.index]

    connections = gpd.GeoDataFrame(geometry=Lines)
    connections.crs = "EPSG:32632"

    intersections = gpd.sjoin(connections, floor_data, predicate='intersects').geometry
    intersections = gpd.GeoDataFrame(geometry=intersections)
    intersections.crs = "EPSG:32632"

    # only leave valid particles in the gdf:
    if len(intersections.geometry) > 0:
        for c in intersections.geometry:
            propagated_particles = propagated_particles.loc[
                propagated_particles['geometry'] != Point(c.coords[1][0], c.coords[1][1])]  # ,'geometry']

    propagated_particles.reset_index(drop=True)
    return  propagated_particles




def check_floor(current_height):
    """
        determining current floor depending on current height

        Parameters
        ----------
        current_height : current calculated height - float

        Returns
        -------
        updated floor number - int
        """
    if current_height <= 2.5 + 1.3:
        return 0

    if current_height > 2.5 + 1.3 and current_height <= 7.5 + 1.3:
        return 1

    if current_height > 7.5 + 1.3 and current_height <= 12.5 + 1.3:
        return 2

    if current_height > 12.5 + 1.3 and current_height <= 17.5 + 1.3:
        return 3

    if current_height > 17.5 + 1.3 and current_height <= 32.5 + 1.3:
        return 4


def set_floor_variables(floor_data,global_rooms,routing_network,transitions,doors):
    current_floor_data = floor_data
    rooms = global_rooms
    temp_routing_lines = routing_network
    current_doors = doors
    current_transition = transitions

    return current_floor_data, rooms, temp_routing_lines,  current_doors, current_transition


def floor_update(step_height, current_height, current_floor, floor_data):
    current_height += step_height
    floor_changed = False
    last_floor = current_floor
    current_floor = check_floor(current_height)


    return current_height, floor_changed, last_floor, current_floor, floor_data