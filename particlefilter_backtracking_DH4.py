#!/usr/bin/env python
# coding: utf-8

# In[1]:


#"normal Backtracking", particle reinitialization with floor change (only particles in connecting stairs or lifts allowed, when floor is changed),
#right now without routing support, but it is possible if desired


# In[2]:


import numpy as np
from numpy.random import normal as randnorm
from numpy.random import uniform as randuniform
from numpy.random import choice as rand_choice
from itertools import accumulate, chain
import operator
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
import math
from numpy import cos as ncos
from numpy import sin as nsin
import time
import random
import geopandas as gpd
import shapely
from shapely.geometry import *


import pandas as pd
import csv


# # Load Data:

# In[3]:


def load_routing(file):
    f = open(file)
    lines = f.readlines()
    f.close()
    
    #appending all lines from the file to List as shapely objects:
    lines_list=[]
    for l in lines:
        if l[0]=='L':

            lines_list.append(shapely.wkt.loads(l.split(',')[0][0:-3]+','+l.split(',')[1][0:-5]+')'))

    gs_lines = gpd.GeoSeries(lines_list)
 
    return gs_lines


# In[4]:


def load_step_data(_trajectory):
    #load the required data for the wanted trajectory
    if _trajectory == 'eight':
        #for eight path:
        SHeading = np.genfromtxt('data/eight/StepHeadigs.csv', delimiter=',', skip_header=0)
        SLengthraw = np.genfromtxt('data/eight/StepLengths.csv', delimiter=',', skip_header=0)
        SLength = SLengthraw 
        ref_eight= np.genfromtxt('data/eight/ref.csv', delimiter=' ', skip_header=0)
        SHeight = np.genfromtxt('data/eight/DeltaHeight.csv', delimiter=' ', skip_header=0)

    if _trajectory == 'zerotofour':
        #for zero2four
        SHeading = np.genfromtxt('data/zero2Four/StepHeadigs.csv', delimiter=',', skip_header=0)
        SLengthraw = np.genfromtxt('data/zero2Four/StepLengths.csv', delimiter=',', skip_header=0)
        SLength = SLengthraw 
        ref_eight= np.genfromtxt('data/zero2Four/Ref_zero2four.csv', delimiter=' ', skip_header=0)
        SHeight = np.genfromtxt('data/zero2Four/deltaheight.csv', delimiter=' ', skip_header=0)

    ###reference points from path: 
    ref_eight_list = []
    for r in ref_eight:
        ref_eight_list.append(Point(r[1],r[2]))

    gs_ref_eight = gpd.GeoSeries(ref_eight_list)
    
    return SHeading,SLength,ref_eight,SHeight,gs_ref_eight


# run the function for the wanted trajectory:

# In[5]:


routing_lines = gpd.read_file("data/routeEG.geojson")
routing_lines = routing_lines.rename(columns={0:'geometry'}).set_geometry('geometry')
routing_lines.crs = "EPSG:32632"

routing_lines1 = gpd.read_file("data/route_1OG.geojson")
routing_lines1 = routing_lines1.rename(columns={0:'geometry'}).set_geometry('geometry')
routing_lines1.crs = "EPSG:32632"

routing_lines4 = gpd.read_file("data/Routing4OG.geojson")
routing_lines4 = routing_lines4.rename(columns={0:'geometry'}).set_geometry('geometry')
routing_lines4.crs = "EPSG:32632"

map_polys = gpd.read_file("data/planEG_polygon_semantic.geojson")
map_polys1 = gpd.read_file("data/plan1OG_polygon_semantic.geojson")
map_polys4 = gpd.read_file("data/Poly-Plan4OGv3-Semantic.geojson")

map_polys.crs = "EPSG:32632"
map_polys1.crs = "EPSG:32632"
map_polys4.crs = "EPSG:32632"

all_walls = map_polys.loc[map_polys['Type'] == 'Wall']
all_walls1 = map_polys1.loc[map_polys1['Type'] == 'Wall']
all_walls4 = map_polys4.loc[map_polys4['Type'] == 'Wall']

only_rooms = map_polys.loc[map_polys['Type'] == 'Room']
only_rooms1 = map_polys1.loc[map_polys1['Type'] == 'Room']
only_rooms4 = map_polys4.loc[map_polys4['Type'] == 'Room']

only_Corridor = map_polys.loc[map_polys['Type'] == 'Corridor']
only_Corridor1 = map_polys1.loc[map_polys1['Type'] == 'Corridor']
only_Corridor4 = map_polys4.loc[map_polys4['Type'] == 'Corridor']

Lift = map_polys.loc[map_polys['Type'] == 'Lift']
Lift1 = map_polys1.loc[map_polys1['Type'] == 'Lift']
Lift4 = map_polys4.loc[map_polys4['Type'] == 'Lift']

only_Stairscase = map_polys.loc[map_polys['Type'] == 'Stairscase']
only_Stairscase1 = map_polys1.loc[map_polys1['Type'] == 'Stairscase']
only_Stairscase4 = map_polys4.loc[map_polys4['Type'] == 'Stairscase']

only_Door = map_polys.loc[map_polys['Type'] == 'Door']
only_Door1 = map_polys1.loc[map_polys1['Type'] == 'Door']
only_Door4 = map_polys4.loc[map_polys4['Type'] == 'Door']


# In[6]:


#the spatial queries are a little quicker when first creating the union of all geometries into one multipolygon:
union_walls = all_walls.geometry.unary_union
union_walls_gdf = gpd.GeoDataFrame(geometry=[union_walls])
union_walls_gdf.crs = "EPSG:32632" 

union_walls1 = all_walls1.geometry.unary_union
union_walls1_gdf = gpd.GeoDataFrame(geometry=[union_walls1])
union_walls1_gdf.crs = "EPSG:32632" 

union_walls4 = all_walls4.geometry.unary_union
union_walls4_gdf = gpd.GeoDataFrame(geometry=[union_walls4])
union_walls4_gdf.crs = "EPSG:32632" 


# In[7]:


walls = [union_walls_gdf,union_walls1_gdf,union_walls4_gdf]
#wallsb = [all_walls,all_walls1,all_walls]


# In[8]:


union_routing_lines = routing_lines.geometry.unary_union

union_routing_lines1 = routing_lines1.geometry.unary_union

union_routing_lines4 = routing_lines4.geometry.unary_union


# In[9]:


#"transition areas" between floors (lifts and staircases)
transition = only_Stairscase.append(Lift)
transition1 = only_Stairscase1.append(Lift1)
transition4 = only_Stairscase4.append(Lift4)


# In[10]:


joined_fur_base_plot = all_walls.geometry.append(only_Door.geometry)
joined_fur_base_plot1 = all_walls1.geometry.append(only_Door1.geometry)
joined_fur_base_plot4 = all_walls4.geometry.append(only_Door4.geometry)

last_room_global = only_Stairscase.append(only_Corridor).append(only_Door).append(only_rooms).append(Lift)
last_room_global1 = only_Stairscase1.append(only_Corridor1).append(only_Door1).append(only_rooms1).append(Lift1)
last_room_global4 = only_Stairscase4.append(only_Corridor4).append(only_Door4).append(only_rooms4).append(Lift4)

global_rooms =[last_room_global,last_room_global1,last_room_global4]


# # Functions:

# # steps and particles

# In[ ]:





# In[11]:


#def initial_particles(firstx, firsty, std, max_num_particles,h_noise,s_noise,current_angle):
def initial_particles(firstx, firsty, std, max_num_particles,h_noise,s_noise):
    """
        creating inital particles
        
        Parameters
        ----------
        firstx : start position x coords - float
        firsty : start position y coords - float
        std : standard deviation for the particle distribution- float
        h_noise : heading noise value - float
        s_noise : step length noise value - float
        max_num_particles : maximum number of particles - int
        current_angle : maximum number of particles - int
        
        Returns
        -------
        initial_particles : initial particles - geopandas geodataframe
        """
    
    initial_particles_geom = []
    Scale = []#error for the steplength
    angle_noise = []#heading error

            
    particle_number = max_num_particles
    #adding the paticles
    for i in range(particle_number):
        initial_particles_geom.append(Point(firstx+randnorm(scale=std), firsty+randnorm(scale=std)))
        Scale.append(randnorm()*s_noise)
        angle_noise.append(randnorm()*h_noise)
        

    #creating the gdf:
    new_particles_df = pd.DataFrame(
    {'Scale': Scale,
     'angle_noise': angle_noise,
     'geometry': initial_particles_geom})
    new_particles_gdf = gpd.GeoDataFrame(new_particles_df, geometry=new_particles_df.geometry)
    new_particles_gdf.crs = "EPSG:32632"
    
    

    
    return new_particles_gdf


# In[12]:


class Step(object):
    #"stepcalss, not all parameters are neccessarily used, but depending on some functionalities they could be handy"
    def __init__(self, _slength, s_heading, _current_floor, _scale=0.0):
        self.length = _slength + _scale#scaling for the step length. 
        self.heading = s_heading 
        #self.height = _height
        self.current_floor = _current_floor
        #self.current_position = _current_position
        self.scale = _scale

        return


# In[13]:


def propagation(_particles,_sheading,_slength):
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
        
        #in principle like the propagation from thomas, but with the same error (Scale) for x and y direction. 
    propagated_geom = [Point(p.coords[0][0] + np.cos(SH+_particles.angle_noise[i])*(SL+ _particles.Scale[i]),p.coords[0][1] + np.sin(SH+_particles.angle_noise[i])*(SL+ _particles.Scale[i])) for i,p in enumerate(_particles.geometry)]
    propagated_scale = [s for s in _particles.Scale]
    propagated_angle = [a for a in _particles.angle_noise]
        

    propagated_particles_df = pd.DataFrame(
    {'Scale': propagated_scale,
     'angle_noise': propagated_angle,
     'geometry': propagated_geom})
    propagated_particles = gpd.GeoDataFrame(propagated_particles_df, geometry=propagated_particles_df.geometry)
    propagated_particles.crs = "EPSG:32632" 

    return propagated_particles


# # checking and backtracking

# In[14]:


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

    if current_height> 2.5+ 1.3 and current_height <= 7.5+ 1.3:
        return  1
    
    if current_height> 7.5+ 1.3 and current_height <= 12.5+ 1.3:
        return 2
    
    if current_height> 12.5+ 1.3 and current_height <= 17.5+ 1.3:
        return 3
    
    if current_height> 17.5+ 1.3 and current_height <= 32.5+ 1.3:
        return 4


# In[15]:


#def CheckNewParticle(_start, _end,floor_data,valid_rooms):
def CheckNewParticle(_start, _end,walls):
    """
        Checking if a wall is between sample particle and new particle
        Parameters
        ----------
        start : position of sample particle - shapely Point geometry
        start : position of new particle - shapely Point geometry
        walls : walls of the building - shapely MultiPolygon geometry object
        
        Returns
        -------
        boolean value: False if wall is in between particles (new particle is invalid), True if not (particle is valid)
        """
    connection = LineString([_start,_end])
    if connection.intersects(walls):

        return False

    return True


# In[16]:


#def BackTrackingParticle(temp_x,temp_y,scale,angle, _step,floor_data, _position_check_number=0, _position_error_max=2.0):
def BackTrackingParticle(temp_x,temp_y,scale,angle, _step,walls):
        """
        actual Backtracking process. a particle position (temp_x,temp_y) is checked if a valid path,
        using the length of the last steps with the error(scale) of the particle, that doesn't cross any walls, 
        can reach this positions.

        Parameters
        ----------
        temp_x : potential x-coordinate of a new particle - float
        temp_y : potential y-coordinate of a new particle - float
        walls : walls of the building - shapely MultiPolygon geometry object
        _step : steps (with heading, length and scale) that are condsidered for the backtracking - list of python object
        scale : potential particle scale for steplength - float
        angle : potential heading deviation of the particle - float
        
        Returns
        -------
        boolean value: True if no intersection with walls, Falls if intersection with walls for one of the steps
        """
        
        last_x,last_y = temp_x,temp_y #suggested position for the new particle
        segments = [(last_x,last_y)]

        for tStep in _step:
            #for every backtracking step (up to 32 steps in reverse) check for the passing
            #of walls:
            last_x -= ncos(tStep.heading+angle)*(tStep.length+scale)
            last_y -= nsin(tStep.heading+angle)*(tStep.length+scale)
            segments.append((last_x,last_y))

        connection = LineString(segments)
        if walls.intersects(connection):

            return False
            
        
        return True


# In[17]:


def BackTracking(new_particle_number,temp_particles,back_tracking_steps,temp_floor_data,try_time_max,h_noise,s_noise): 
    """
        Backtracking. New particles are randomly created around axisting particles (the sampling can be done weighted). The new particles
        are propagated backwords and checked for intersections with walls.

        Parameters
        ----------
        temp_particles : remaining particles after propagation and checking - geopandas geodataframe
        temp_floor_data : walls of the building - shapely geometry
        back_tracking_steps : steps (with heading, length and scale) that are condsidered for the backtracking - python object
        h_noise : heading noise - float
        s_noise : step length noise - float
        try_time_max : max number of tries per particle sample - int
        new_particle_number : max number of newly generated particles - int
        
        Returns
        -------
        backtracking_particles_gdf: particles including the original temp_particles as well as the newly generated particles - geopandas geodataframe
        """
    
    walls = temp_floor_data.geometry[0]
    generate_radius_min,generate_radius_max = 0.1, 1.0 #random radius around sampled particle

    temp_geoms = list(temp_particles.geometry)
    scale_list = list(temp_particles.Scale)
    angle_list = list(temp_particles.angle_noise)

    
    point_samples = np.array(rand_choice(np.array(temp_particles.geometry),new_particle_number))
    
    for n in range(len(point_samples)):  

            try_time = try_time_max

            while try_time > 0:#per particle only a certain amount of tries are allowed before passing on to evoid too long loop times

                temp_angle,temp_radius = randuniform(-np.pi, np.pi),randuniform(generate_radius_min, generate_radius_max)
                new_x,new_y = np.add(point_samples[n].coords[0][0], np.multiply(ncos(temp_angle), temp_radius)),  np.add(point_samples[n].coords[0][1], np.multiply(nsin(temp_angle), temp_radius))#determine position of new particle around sample particle

                #new particle angle and steplength error:
                Scale= randnorm()*s_noise
                angle= randnorm()*h_noise
                
                #check for validity of new particle:
                if CheckNewParticle((point_samples[n].coords[0][0], point_samples[n].coords[0][1]), (new_x, new_y), walls):
                    
                    #check the backtracking steps:
                    if BackTrackingParticle(new_x,new_y,Scale,angle,back_tracking_steps,walls):

                        scale_list.append(Scale)
                        angle_list.append(angle)
                        temp_geoms.append(Point(new_x,new_y))
                        break
                try_time -= 1
                
            new_particle_number -= 1

        
    #add all valid new particle to existing particles:
    new_particles_df = pd.DataFrame(
    {
     'Scale': scale_list,
     'angle_noise': angle_list,
     'geometry': temp_geoms})
    backtracking_particles_gdf = gpd.GeoDataFrame(new_particles_df, geometry=new_particles_df.geometry)
    backtracking_particles_gdf.crs = "EPSG:32632" 
    return backtracking_particles_gdf


# In[18]:


#def CheckParticleSteps(_particles, _step,floor_data,floor_changed,current_transition,valid_rooms):
def CheckParticleSteps(_particles, _step,floor_data,floor_changed,current_transition):
    """
        check the propagation of the particles for the current step for their validity (invalid if walls are crosssed).
        When floor has been changed only particles in transition zones (stairs and lifts) between current and last floor are valid

        Parameters
        ----------
        _particles : particles (including position, scale and angle) - geopandas geodataframe
        floor_data : walls of the building - shapely geometry
        floor_changed : if floor has been changed or not - boolean
        current_transition : current transition zones (stairs and lifts) - geopandas geodataframe
        
        
        Returns
        -------
        valid_particles : valid propagated particles - geopandas geodataframe
        """
    
    if floor_changed:
        
        _particles = gpd.overlay(_particles, current_transition, how='intersection')
        _particles.reset_index(drop=True)
        
    
    #check for crossing of walls:
    propagated_particles = propagation(_particles,_step.heading,_step.length)
    
    Lines = [LineString([(_particles.geometry[i].coords[0]),(p.coords[0])]) for i,p in enumerate(propagated_particles.geometry)]

    connections = gpd.GeoDataFrame(geometry=Lines)
    connections.crs = "EPSG:32632" 
  
    intersections = gpd.sjoin(connections,floor_data, predicate='intersects')#.geometry
    intersections.crs = "EPSG:32632"
    valid_particles = gpd.overlay(propagated_particles, intersections, how='difference')
            
    valid_particles.reset_index(drop=True)
    return valid_particles


# In[19]:


def weighting_by_routing(particles_gdf,routing_lines,R_rout): #layer of routing graph not its geometry
    """
        weighting by route. particles near to the route will get more weights base on normal distribution

        Parameters
        ----------
        propagated_particles : particle coordinates for the current step - geopandas type
        room : the selected polygon - geopandas type
        
        Returns
        -------
        w: weights - numpy array of "num" number of float weights
        """
    w = []
    
    particles_gdf.reset_index(drop=True)
    for point in particles_gdf.geometry:
        orthogonal_dist = np.min(routing_lines.distance(point))
        
        if orthogonal_dist < 3:
            w.append(np.exp(-0.5*orthogonal_dist * (1/R_rout) * orthogonal_dist)) ####R_rout is not defined
        else:
            w.append(0.001)
    return np.array(w)


# In[ ]:





# In[20]:


def resampling(w,propagated_particles,eff):
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
    
    top=[]
    
        
    top.append(w[0])
    for i in range(len(propagated_particles.geometry)-1):
        top.append(top[i] + w[i+1])


    resampled_geom = []
    resampled_scale = []
    resampled_angle = []
    for i in range(len(propagated_particles.geometry)):
        zf = np.random.rand(1)  #gleichverteilte Zufallszahl ziehen

        for jj in propagated_particles.index:
            if top[jj]>zf:
                new_x = propagated_particles.geometry[jj].coords[0][0]
                new_y = propagated_particles.geometry[jj].coords[0][1]
                temp_angle,temp_radius = randuniform(-math.pi, math.pi),randuniform(0.1, 0.5)#random radius around particle for resampling (like in backtracking)
    
                resampled_geom.append(Point(new_x + ncos(temp_angle) * temp_radius,new_y + nsin(temp_angle) * temp_radius))
                resampled_scale.append(propagated_particles.Scale[jj])
                resampled_angle.append(propagated_particles.angle_noise[jj])
                break 

    resampled_particles_gdf = pd.DataFrame(
    {
     'Scale': resampled_scale,
     'angle_noise': resampled_angle,
     'geometry': resampled_geom})
    new_particles_gdf = gpd.GeoDataFrame(resampled_particles_gdf, geometry=resampled_particles_gdf.geometry)
    new_particles_gdf.crs = "EPSG:32632"
    
    return new_particles_gdf
        
   


# logging of data:

# In[21]:


def weighted_avg_and_std_geopoints(points_layer,weights):
    w_sum_x = 0
    w_sum_y = 0
    xvalues = np.array([])
    yvalues = np.array([])

    for i,points in points_layer.iterrows():
        xvalues = np.append(xvalues, points.geometry.coords[0][0])
        yvalues = np.append(yvalues, points.geometry.coords[0][1])

       
    x_average = np.average(xvalues, weights=weights)
    y_average = np.average(yvalues, weights=weights)
    x_variance = np.average((xvalues-x_average)**2, weights=weights)
    y_variance = np.average((yvalues-y_average)**2, weights=weights)
    
    return Point(x_average,y_average),math.sqrt(x_variance),math.sqrt(y_variance)


# # main

# In[22]:




# In[23]:


#x_0 = ref_eight[0,1]
#y_0 = ref_eight[0,2]
run_times = []
run_times1 = []
run_times2 = []


# In[ ]:





# ### run function

# this function runs the mapmatching process

# In[24]:


def run(_trajectory,walls):
    """
    mapmatching and backtracking process for each step. Resampling and weighting by routing can be uncommented 
    or left inactive. The live plotting could also be deactivated by commenting it out to increase performance slightly

    Parameters
    ----------
    _trajectory : choose the trajectory - string
    walls : walls of the building - list of geopandas GeoDataFrames


    Returns
    -------
    positions_list : x and y coordinates and height value for each step - list of lists (of floats)
    std_list : the positions standard deviation for each step - list of floats
    trajectory : list of the positions for each step - list of shapely point geometries
    """
    
    positions_list = []
    std_list = []
    trajectory = []
    
    userheight = 1.7
    
    #parameter dependent on the given trajectory:
    if _trajectory == 'eight':
        #eight-path:
        SHeading,SLength,ref_eight,SHeight,gs_ref_eight = load_step_data('eight')
        heading0 = 200 * np.pi/180
        start_height = 26 + .8*userheight
        current_floor = 4
        step_scale = 0
        base = all_walls4
        
    elif _trajectory == 'zerotofour':
        #zero2four:
        SHeading,SLength,ref_eight,SHeight,gs_ref_eight = load_step_data('zerotofour')
        heading0 = 20 * np.pi/180
        start_height = 0 + .8*userheight#26 + .8*userheight
        current_floor = 0
        step_scale = 0
        base = all_walls
    
    x_0 = ref_eight[0,1]
    y_0 = ref_eight[0,2]
    current_height = start_height
    temp_step = []
    max_num_particles = 200
    std=1 #standard deviation for initial particle position

    R_rout = 1.5**2# noise for routing support
    h_noise = 15*np.pi/180   #heading noise, not used for the current propagation
    s_noise = 0.1  #step length noise, not used for the current propagation

    start = Point(x_0,y_0) #startposition
    
    calculated_position= start
    


    ##initial Particles (befor first step):
    start_particles = initial_particles(x_0, y_0, std, max_num_particles,h_noise,s_noise)

    ## all particles each step
    particles = start_particles
    steps = []
    #step_scale = 0.2

    floor_data = walls
    
     ##plot: 


    fig, ax= plt.subplots(figsize=(20,15))
    plt.ion()
    base.plot(ax = ax, color='black')

    frame = fig.text(0.2, 0.8, "", size=9,
                     ha="left", va="top",
                     )
    fig.canvas.draw()
    fig.show()
    plt.pause(0.01)
    

    last_transition = transition
    current_transition = transition
    everything_start_time = time.time()
    
    #duration = [i for i in range(len(SHeading))]
    for s in range(len(SHeading)):

        start_time_all = time.time()
        
        #calculating height and checking for current floor and floor change:
        
        current_height += SHeight[s]
        floor_changed = False
        last_floor = current_floor
        current_floor = check_floor(current_height)
        

        if current_floor != last_floor:
            floor_changed = True
            #particles = initial_particles(calculated_position.coords[0][0], calculated_position.coords[0][1], 0, max_num_particles,h_noise,s_noise)
            last_transition = current_transition
            
        if current_floor == 0:
            temp_floor_data = floor_data[0]
            valid_rooms = global_rooms[0]
            temp_routing_lines = routing_lines
            temp_transition = gpd.overlay(transition, last_transition, how='intersection')
            current_transition = transition
            
        elif current_floor == 1:
            temp_floor_data = floor_data[1]
            valid_rooms = global_rooms[1]
            temp_routing_lines = routing_lines1
            temp_transition = gpd.overlay(transition1, last_transition, how='intersection')
            current_transition = transition1
            
        elif current_floor == 2 or current_floor == 3:
            temp_floor_data = floor_data[1]
            valid_rooms = transition4
            #temp_routing_lines = routing_lines1
            temp_transition = gpd.overlay(transition4, last_transition, how='intersection')
            current_transition = transition4
            
        elif current_floor == 4:
            temp_floor_data = floor_data[2]
            valid_rooms = global_rooms[2]
            temp_routing_lines = routing_lines4
            temp_transition = gpd.overlay(transition4, last_transition, how='intersection')
            current_transition = transition4

        #creating step object
        temp_step = Step(SLength[s], heading0 + SHeading[s],current_floor, _scale=step_scale)


        start_time = time.time()
       
        #propagate all particles and delete all invalid particles:
        temp_particles = CheckParticleSteps(particles,temp_step,temp_floor_data,floor_changed,temp_transition)
        
        res_particles_geom = temp_particles.geometry

        section1_time = time.time() - start_time
        run_times1.append(section1_time)
        
        steps.append(temp_step)
        
        #backtracking parameter:
        particle_number = len(temp_particles.geometry)
        try_time_max = 8
        back_tracking_step_number = min(32, s) + 1
        
        #weighting (if enabled):
        
        w0 = np.full((particle_number), 1/particle_number)
        #w_r = weighting_by_routing(temp_particles,temp_routing_lines,R_rout)
        #w = w_r*w0
        w = w0
        
        w = np.divide(w,np.sum(w))
        
        
        ###standarddeviation,logging and calculation of estimated position:
        calculated_position,x_std,y_std = weighted_avg_and_std_geopoints(temp_particles,w)
        trajectory.append(calculated_position)#calculated position for this step
        
        positions_list.append((calculated_position.coords[0][0],calculated_position.coords[0][1],current_height))
        std_list.append((x_std,y_std,0))
        
        
        #step correction calculator (used when floor is changed):
        temp_step_scale = np.mean(temp_particles.Scale)
        if floor_changed:
            #print(temp_step_scale)
            step_scale += temp_step_scale
        
        ###########resampling:###################################
        
        eff = 1/np.sum(w**2)
        #if eff <0.8*particle_number:
            #temp_particles = resampling(w,temp_particles,eff)


        ############backtracking:###################################

        new_particle_number = max_num_particles - particle_number
        
        if back_tracking_step_number == 1:
            back_tracking_steps = [steps[-1]] #neccessary for first step in eight path, because "for b in range(1,1) doesn't work"
        else:
            back_tracking_steps = [steps[-b] for b in range(1, back_tracking_step_number) if steps[-b].current_floor == current_floor]
        
        #start_time_2 = time.time()

        #generating new particles after checking the backtracking of the particles
        particles = BackTracking(new_particle_number,temp_particles,back_tracking_steps,temp_floor_data,try_time_max,h_noise,s_noise)
                                
        
        #section2_time = time.time() - start_time_2
        
        #run_times2.append(section2_time)

        time_all = time.time() - start_time_all
        run_times.append(time_all)
        
        ########plot:
        ax.clear()
        
        if current_floor == 0:
            joined_fur_base_plot.plot(ax = ax, color='black')
            routing_lines.plot(ax = ax, color='blue')
            transition.plot(ax = ax, color='grey')

        elif current_floor == 1:
            joined_fur_base_plot1.plot(ax = ax, color='black')
            routing_lines1.plot(ax = ax, color='blue')
            transition1.plot(ax = ax, color='grey')
            
        elif current_floor == 2 or current_floor == 3:
            temp_transition.plot(ax = ax, color='grey')

        elif current_floor == 4:
            joined_fur_base_plot4.plot(ax = ax, color='black')   
            routing_lines4.plot(ax = ax, color='blue')
            transition4.plot(ax = ax, color='grey')

        gpd.GeoSeries(res_particles_geom).plot(ax = ax, color='green',markersize=1)
        
        gpd.GeoSeries(calculated_position).plot(ax = ax, color='blue', markersize=50)

        Artist.remove(frame)
        textstr = '\n'.join((f'Step: {s}',
        f'current_floor: {current_floor}',
        #r'x-coords=%.2f' % (propagated_particles_x_std ),
        f'valid particles: {particle_number}',
        f'Height: {current_height:.2f}',
        #r'particles w0: %.2f' % (len(particles_w0))
        #f'eff: {eff:.2f}',
        #f'0.8_p: {0.8*particle_number:.2f}',
        f'time whole: {time_all:.2f}',
        f'step scale: {step_scale:.2f}',
        #f'part. Scale: {np.mean(temp_particles.Scale):.2f}',
        f'current floor: {current_floor}',
        #f'time 1st part: {section1_time:.2f}',
        #f'time 2nd part: {section2_time:.2f}'
                            ))
        frame = fig.text(0.2, 0.6, textstr, size=9,
                     ha="left", va="top",
                     )

        Artist.set_visible(frame, True)
        # To remove the artist

        fig.canvas.draw()
        plt.pause(0.1)
        #plt.savefig(f"PF_backtracking_024_{s}.jpg")
        
    print("---Whole Tracjectory %s seconds ---" % (time.time() - everything_start_time ))

    return positions_list, std_list, trajectory
 


# ## Plot functions:

# In[25]:



positions_list, std_list, trajectory = run('eight',walls)
#positions_list, std_list, trajectory = run('zerotofour',walls)


# # Plotting:

# ### static plot:

# In[46]:




base = all_walls4.plot(color='black', edgecolor='black',figsize=(10,10))
#gpd.GeoSeries(gs_particles[108]).plot(ax=base, marker='o', color='green', markersize=1)
#gpd.GeoSeries(gs_particles_w0[108]).plot(ax = base, color='red',markersize=5)
routing_lines4.plot(ax = base,linewidth=1, color='blue')
gpd.GeoSeries(trajectory).plot(ax=base, marker='o', color='green', markersize=10,label = 'Trajectory')
#gs_route.plot(ax=base,color='green', linewidth=4)
#gs_ref_eight.plot(ax=base, marker='o', color='red', markersize=1)
#gs_route_pts.plot(ax=base, marker='o', color='blue', markersize=4)
#gpd.GeoSeries(active_r_line[54]).plot(ax = base,color = 'red')
#gpd.GeoSeries(heading_i[52]).plot(ax = base,color = 'red')
plt.savefig(f"BT_floorcheck_andstepscale_update_0cm_024_1.jpg")

plt.show()


# In[30]:




base = all_walls4.plot(color='black', edgecolor='black',figsize=(10,10))
#gpd.GeoSeries(gs_particles[108]).plot(ax=base, marker='o', color='green', markersize=1)
#gpd.GeoSeries(gs_particles_w0[108]).plot(ax = base, color='red',markersize=5)
routing_lines4.plot(ax = base,linewidth=1, color='blue')
gpd.GeoSeries(trajectory).plot(ax=base, marker='o', color='green', markersize=10,label = 'Trajectory')
#gs_route.plot(ax=base,color='green', linewidth=4)
#gs_ref_eight.plot(ax=base, marker='o', color='red', markersize=1)
#gs_route_pts.plot(ax=base, marker='o', color='blue', markersize=4)
#gpd.GeoSeries(active_r_line[54]).plot(ax = base,color = 'red')
#gpd.GeoSeries(heading_i[52]).plot(ax = base,color = 'red')
plt.savefig(f"BT_floorcheck_andstepscale_update_0cm_024_8.jpg")

plt.show()


# In[36]:


# writing into file:  
filename_run = 'output/eight/BT_floorcheck_0cm_024.csv'
with open(filename_run, mode='w') as run_file:

    run_file_writer = csv.writer(run_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    #run_file_writer.writerow(['all particles that are not in transition zones are deleted with floor change'])
    run_file_writer.writerow(['X','Y','Z','X_std','Y_std','Z_std'])
    
    for l in range(len(positions_list)):

        run_file_writer.writerow([str(positions_list[l][0]),str(positions_list[l][1]),str(positions_list[l][2]),str(std_list[l][0]),str(std_list[l][1]),str(std_list[l][2])])





