import geopandas as gpd
import numpy as np
import pandas as pd
from numpy import cos as ncos
from numpy import sin as nsin
from numpy.random import choice as rand_choice
from numpy.random import normal as randnorm
from numpy.random import uniform as randuniform
from shapely.geometry import *


# line of sight conept

def CheckNewParticle_LOS(_start, _end, walls):
    """
        Checking if a wall is between sample particle and new particle
        Parameters
        ----------
        _start : position of sample particle - shapely Point geometry
        _end : position of new particle - shapely Point geometry
        walls : walls of the building - shapely MultiPolygon geometry object

        Returns
        -------
        boolean value: False if wall is in between particles (new particle is invalid), True if not (particle is valid)
        """
    connection = LineString([_start, _end])
    if connection.intersects(walls):
        return False

    return True


def BackTrackingParticle_LOS(temp_x, temp_y, angle, length_noise, _step, walls):
    """
    actual Backtracking process. a particle position (temp_x,temp_y) is checked if a valid path,
    using the length of the last steps with the error(length_noise) of the particle, that doesn't cross any walls,
    can reach this positions.

    Parameters
    ----------
    temp_x : potential x-coordinate of a new particle - float
    temp_y : potential y-coordinate of a new particle - float
    walls : walls of the building - shapely MultiPolygon geometry object
    _step : steps (with heading, length and scale) that are condsidered for the backtracking - list of python object
    length_noise : potential particle length_noise for steplength - float
    angle : potential heading deviation of the particle - float

    Returns
    -------
    boolean value: True if no intersection with walls, Falls if intersection with walls for one of the steps
    """

    last_x = temp_x  # suggested position for the new particle
    last_y = temp_y
    segments = [(last_x, last_y)]

    for tStep in _step:
        # for every backtracking step (up to 32 steps in reverse) check for the passing
        # of walls:
        last_x -= ncos(tStep.heading + angle) * (tStep.length + length_noise)
        last_y -= nsin(tStep.heading + angle) * (tStep.length + length_noise)
        segments.append((last_x, last_y))

    connection = LineString(segments)
    if walls.intersects(connection):
        return False

    return True


# klasse!
def BackTracking_LOS(new_particle_number, temp_particles, back_tracking_steps, floor_data, try_time_max, h_noise,
                     s_noise):
    """
    Backtracking. New particles are randomly created around axisting particles (the sampling can be done weighted). The new particles
    are propagated backwords and checked for intersections with walls.

    Parameters
    ----------
    temp_particles : remaining particles after propagation and checking - geopandas geodataframe
    floor_data : walls of the building - shapely geometry
    back_tracking_steps : steps (with heading, length and scale) that are condsidered for the backtracking - python object
    h_noise : heading noise - float
    s_noise : step length noise - float
    try_time_max : max number of tries per particle sample - int
    new_particle_number : max number of newly generated particles - int

    Returns
    -------
    backtracking_particles_gdf: particles including the original temp_particles as well as the newly generated particles - geopandas geodataframe
    """

    walls = floor_data.geometry[0]
    generate_radius_min = 0.1
    generate_radius_max = 1.0  # random radius around sampled particle
    geoms_list = list(temp_particles.geometry)
    length_noise_list = list(temp_particles.length_noise)
    angle_list = list(temp_particles.angle_noise)
    point_samples = np.array(rand_choice(np.array(temp_particles.geometry), new_particle_number))
    try_time_max = try_time_max

    for n in range(len(point_samples)):

        try_time = try_time_max

        while try_time > 0:  # per particle only a certain amount of tries are allowed before passing on to evoid too long loop times

            temp_angle = randuniform(-np.pi, np.pi)
            temp_radius = randuniform(generate_radius_min, generate_radius_max)
            new_x, new_y = np.add(point_samples[n].coords[0][0], np.multiply(ncos(temp_angle), temp_radius)), np.add(
                point_samples[n].coords[0][1],
                np.multiply(nsin(temp_angle), temp_radius))  # determine position of new particle around sample particle

            # new particle angle and steplength error:
            length_noise = randnorm() * s_noise
            angle = randnorm() * h_noise

            # check for validity of new particle:
            if CheckNewParticle_LOS((point_samples[n].coords[0][0], point_samples[n].coords[0][1]), (new_x, new_y),
                                    walls):

                # check the backtracking steps:
                if BackTrackingParticle_LOS(new_x, new_y, length_noise, angle, back_tracking_steps, walls):
                    length_noise_list.append(length_noise)
                    angle_list.append(angle)
                    geoms_list.append(Point(new_x, new_y))
                    break
            try_time -= 1

        # new_particle_number -= 1

    # add all valid new particle to existing particles:
    new_particles_df = pd.DataFrame(
        {
            'length_noise': length_noise_list,
            'angle_noise': angle_list,
            'geometry': geoms_list})
    backtracking_particles_gdf = gpd.GeoDataFrame(new_particles_df, geometry=new_particles_df.geometry)
    backtracking_particles_gdf.crs = "EPSG:32632"
    return backtracking_particles_gdf


def CheckNewParticle_rooms(_start, _end, valid_rooms):
    """
    checking new particles for their validity, returns False if the new particle is not in a valid room (e.g. the current room)
    no checking for doors here, but could also be implemented here
    or if a wall is between the new particle and the sample particle
    Parameters
    ----------
    _start : position of sample particle - shapely Point geometry
    _end : position of new particle - shapely Point geometry
    valid_rooms : all rooms currently defined as valid - MultiPolygon-Geometry

    Returns
    -------
    Boolean Value (True or False)
    """

    if any(valid_rooms.contains(Point(_end))):
        return True

    return False


def BackTrackingParticle_rooms(step, temp_x, temp_y, angle, length_noise, valid_rooms, doors, current_transition):
    """
    this is the actual Backtracking process. a particle position (temp_x,temp_y) is checked if a valid path,
    using the length of the last steps with the error(scale) of the particle, that only goes through the valid_rooms at each step,
    can reach this position

    Parameters
    ----------
    temp_x : potential x-coordinate of a new particle - float
    temp_y : potential y-coordinate of a new particle - float
    step : steps (with heading, length and scale) that are condsidered for the backtracking - list of python object
    length_noise : potential particle length_noise for steplength - float
    angle : potential heading deviation of the particle - float
    valid_rooms : all rooms currently defined as valid - MultiPolygon-Geometry
    doors : doors of current floor - MultiPolygon-Geometry
    current_transition : currently valid Stairs and Lifts - MultiPolygon-Geometry


    Returns
    -------
    Boolean Value (True or False)
    """

    # last_x, last_y = temp_x, temp_y  # suggested position for the new particle

    # for every backtracking step (up to 32 steps in reverse) calculate displacement:
    for tStep in step:

        last_x = temp_x - ncos(tStep.heading + angle) * (tStep.length + length_noise)
        last_y = temp_y - nsin(tStep.heading + angle) * (tStep.length + length_noise)

        temp_position = Point(last_x, last_y)  # particle position at this step

        if tStep.valid_rooms.contains(temp_position):  # test if particles are in valid rooms for the step
            # this includes the current room of the steps position and
            # and all rooms that have been accessed by other particles through doors
            temp_x, temp_y = last_x, last_y
            continue

        if LineString([(last_x, last_y), (temp_x, temp_y)]).intersects(
                doors):  # if not but crosses door, it is also valid

            temp_x, temp_y = last_x, last_y
            continue

        else:
            return False

    return True


def BackTracking_rooms(new_particle_number, temp_particles, back_tracking_steps, try_time_max,
                       h_noise, s_noise, valid_rooms, doors, current_transition):
    """
    Backtracking. New particles are randomly created around axisting particles (the sampling can be done weighted). The new particles
    are propagated backwards and checked if they are in a valid correct room.
    Parameters
    ----------
    temp_particles : remaining particles after propagation and checking - geopandas geodataframe
    back_tracking_steps : steps (with heading, length and scale) that are condsidered for the backtracking - python object
    try_time_max : max number of tries per particle sample - int
    new_particle_number : max number of newly generated particles - int
    h_noise : heading noise - float
    s_noise : step length noise - float
    valid_rooms : all rooms currently defined as valid - MultiPolygon-Geometry
    doors : doors of current floor - MultiPolygon-Geometry
    current_transition : currently valid Stairs and Lifts - MultiPolygon-Geometry

    Returns
    -------
    backtracking_particles_gdf : particles including the original temp_particles as well as the newly generated particles - geopandas geodataframe
    """
    rooms = valid_rooms  # temp_floor_data#.geometry[0]
    generate_radius_min, generate_radius_max = 0.2, 2.0  # random radius around sampled particle

    temp_geoms = list(temp_particles.geometry)
    length_noise_list = list(temp_particles.length_noise)
    angle_list = list(temp_particles.angle_noise)

    point_samples = np.array(rand_choice(np.array(temp_particles.geometry), new_particle_number))

    x_samples = [p.coords[0][0] for p in point_samples]
    y_samples = [p.coords[0][1] for p in point_samples]

    for n in range(len(point_samples)):

        # from the sample particle take position:
        # angle_sample = angle_samples[n]
        try_time = try_time_max

        while try_time > 0:  # per particle only a certain amount of tries are allowed before passing on to evoid too long loop times

            # for t in range(try_time_max):
            temp_angle, temp_radius = randuniform(-np.pi, np.pi), randuniform(generate_radius_min, generate_radius_max)
            new_x, new_y = np.add(x_samples[n], np.multiply(ncos(temp_angle), temp_radius)), np.add(y_samples[n],
                                                                                                    np.multiply(nsin(
                                                                                                        temp_angle),
                                                                                                        temp_radius))  # determine position of new particle around sample particle

            # new particle angle and steplength error:
            length_noise = randnorm() * s_noise
            angle = randnorm() * h_noise

            # check for validity of new particle:
            # if CheckNewParticle_rooms((x_samples[n], y_samples[n]), (new_x, new_y), rooms):

            # check the backtracking steps:
            if BackTrackingParticle_rooms(back_tracking_steps, new_x, new_y, angle, length_noise, rooms, doors,
                                          current_transition):
                length_noise_list.append(length_noise)
                angle_list.append(angle)
                temp_geoms.append(Point(new_x, new_y))
                break
            try_time -= 1

        new_particle_number -= 1

    # add all valid new particle to existing particles:
    new_particles_df = pd.DataFrame(
        {
            'length_noise': length_noise_list,
            'angle_noise': angle_list,
            'geometry': temp_geoms})
    backtracking_particles_gdf = gpd.GeoDataFrame(new_particles_df, geometry=new_particles_df.geometry)
    backtracking_particles_gdf.crs = "EPSG:32632"
    return backtracking_particles_gdf


# routing concept:


def CheckNewParticle_routing(particle, routing_lines):
    """
    checking new particles for their validity, returns False if the new particle is too far away from the next routing edge.
    Parameters
    ----------
    particle : position of a potential new particle - Shapely Point-Geometry
    routing_lines : edges of the routing network - Shapely MultiLineString-Geometry

    Returns
    -------
    Boolean Value (True or False)
    """
    orthogonal_dist = np.min(routing_lines.distance(particle))
    if orthogonal_dist > 2:  # this is the maximal allowed distance to the routing edge
        return False
    return True


def BackTrackingParticle_routing(temp_x, temp_y, length_noise, angle, _step, routing_lines):
    """
    this is the actual Backtracking process. a particle position (temp_x,temp_y) is checked if a valid path,
    using the length of the last steps with the error(scale) of the particle, that doesn't go too far from any routing edge,
    can reach this position
    
    Parameters
    ----------
    temp_x : potential x-coordinate of a new particle - float
    temp_y : potential y-coordinate of a new particle - float
    _step : steps (with heading, length and scale) that are condsidered for the backtracking - list of python object
    length_noise : potential particle length_noise for steplength - float
    angle : potential heading deviation of the particle - float
    routing_lines : edges of the routing network - Shapely MultiLineString-Geometry
    
    Returns
    -------
    Boolean Value (True or False)
    """

    last_x, last_y = temp_x, temp_y  # suggested position for the new particle

    displacement_x = [ncos(tStep.heading + angle) * (tStep.length + length_noise) for tStep in _step]
    displacement_y = [nsin(tStep.heading + angle) * (tStep.length + length_noise) for tStep in _step]

    # backtracking_points = [Point(ncos(tStep.heading+angle)*tStep.length+scale,nsin(tStep.heading+angle)*tStep.length+scale) for tStep in _step]
    dl = [i for i in range(len(displacement_x))]
    for d in dl:
        last_x -= displacement_x[d]
        last_y -= displacement_y[d]

        orthogonal_dist = np.min(routing_lines.distance(Point(last_x, last_y)))

        if orthogonal_dist > 2:  # maximum distance allowed to touring edge
            return False

    return True


def BackTracking_routing(new_particle_number, temp_particles, back_tracking_steps, try_time_max, h_noise, s_noise,
                         routing_lines):
    """
    Backtracking. New particles are randomly created around axisting particles (the sampling can be done weighted). The new particles
    are propagated backwords and checked for distance to routing edge.
    
    Parameters
    ----------
    routing_lines : routing edges - shapely geometry
    temp_particles : remaining particles after propagation and checking - geopandas geodataframe
    back_tracking_steps : steps (with heading, length and scale) that are condsidered for the backtracking - python object
    try_time_max : max number of tries per particle sample - int
    new_particle_number : max number of newly generated particles - int
    h_noise : heading noise - float
    s_noise : step length noise - float

    Returns
    -------
    backtracking_particles_gdf : particles including the original temp_particles as well as the newly generated particles - geopandas geodataframe
    """
    generate_radius_min, generate_radius_max = 0.2, 2.0  # random radius around sampled particle

    temp_geoms = list(temp_particles.geometry)
    length_noise_list = list(temp_particles.length_noise)
    angle_list = list(temp_particles.angle_noise)

    point_samples = np.array(rand_choice(np.array(temp_particles.geometry), new_particle_number))
    x_samples = [p.coords[0][0] for p in point_samples]
    y_samples = [p.coords[0][1] for p in point_samples]

    # while new_particle_number > 0:#while the current particle number is smaller than the max particle number, suggest new particles

    for n in range(len(point_samples)):

        temp_angle, temp_radius = randuniform(-np.pi, np.pi, try_time_max), randuniform(generate_radius_min,
                                                                                        generate_radius_max,
                                                                                        try_time_max)
        new_x, new_y = np.add(x_samples[n], np.multiply(ncos(temp_angle), temp_radius)), np.add(y_samples[n],
                                                                                                np.multiply(
                                                                                                    nsin(temp_angle),
                                                                                                    temp_radius))

        # while try_time > 0:#per particle only a certain amount of tries are allowed before passing on to evoid too long loop times

        for t in range(try_time_max):

            # new particle angle and steplength error:
            length_noise = randnorm() * s_noise
            angle = randnorm() * h_noise

            # check for validity of new particle:
            if CheckNewParticle_routing(Point(new_x[t], new_y[t]), routing_lines):

                # check the backtracking steps:
                if BackTrackingParticle_routing(new_x[t], new_y[t], length_noise, angle, back_tracking_steps,
                                                routing_lines):
                    length_noise_list.append(length_noise)
                    angle_list.append(angle)
                    temp_geoms.append(Point(new_x[t], new_y[t]))
                    break

        new_particle_number -= 1

    # add all valid new particle to existing particles:
    new_particles_df = pd.DataFrame(
        {
            'length_noise': length_noise_list,
            'angle_noise': angle_list,
            'geometry': temp_geoms})
    backtracking_particles_gdf = gpd.GeoDataFrame(new_particles_df, geometry=new_particles_df.geometry)
    backtracking_particles_gdf.crs = "EPSG:32632"
    return backtracking_particles_gdf
