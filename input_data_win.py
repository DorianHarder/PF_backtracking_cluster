import geopandas as gpd
import numpy as np
import shapely
import shapely.wkt
from shapely.geometry import *
# gpd.options.use_pygeos = True


def load_routing(file):
    """

    Parameters
    ----------
    file: path of csv file

    Returns
    -------
    gs_lines: geoseries of shapely LineString geometries

    """
    f = open(file)
    lines = f.readlines()
    f.close()

    # appending all lines from the file to List as shapely objects:
    lines_list = []
    for l in lines:
        if l[0] == 'L':
            lines_list.append(shapely.wkt.loads(l.split(',')[0][0:-3] + ',' + l.split(',')[1][0:-5] + ')'))

    gs_lines = gpd.GeoSeries(lines_list)

    return gs_lines


def Load_Step_Data(_trajectory):
    """

    Parameters
    ----------
    _trajectory: name of the trajectory, as string

    Returns
    -------
    SHeading: np.array of floats, values for step heading
    SLength:  np.array of floats, values for step length
    ref_eight: np.array of tuples containing a pair of floats, coordinates of ref. points
    SHeight: np.array of floats, values for height differences
    gs_ref_eight: geoseries of shapely point geometries, reference points

    """
    # load the required data for the wanted trajectory
    if _trajectory == 'eight':
        # for eight path:
        SHeading = np.genfromtxt('data\eight\StepHeadigs.csv', delimiter=',', skip_header=0)
        SLengthraw = np.genfromtxt('data\eight\StepLengths.csv', delimiter=',', skip_header=0)
        SLength = SLengthraw
        ref = np.genfromtxt('data\eight\ref.csv', delimiter=' ', skip_header=0)
        SHeight = np.genfromtxt('data\eight\DeltaHeight.csv', delimiter=' ', skip_header=0)

    if _trajectory == 'zerotofour':
        # for zero2four
        SHeading = np.genfromtxt('data\zero2Four\StepHeadigs.csv', delimiter=',', skip_header=0)
        SLengthraw = np.genfromtxt('data\zero2Four\StepLengths.csv', delimiter=',', skip_header=0)
        SLength = SLengthraw
        ref = np.genfromtxt('data\zero2Four\Ref_zero2four.csv', delimiter=' ', skip_header=0)
        SHeight = np.genfromtxt('data\zero2Four\deltaheight.csv', delimiter=' ', skip_header=0)

    ###reference points from path:
    ref_list = []
    for r in ref:
        ref_list.append(Point(r[1], r[2]))

    gs_ref = gpd.GeoSeries(ref_list)

    return SHeading, SLength, ref, SHeight, gs_ref


routing_lines = gpd.read_file("data/routeEG.geojson")
routing_lines = routing_lines.rename(columns={0: 'geometry'}).set_geometry('geometry')
routing_lines.crs = "EPSG:32632"

routing_lines1 = gpd.read_file("data/route_1OG.geojson")
routing_lines1 = routing_lines1.rename(columns={0: 'geometry'}).set_geometry('geometry')
routing_lines1.crs = "EPSG:32632"

routing_lines4 = gpd.read_file("data/Routing4OG.geojson")
routing_lines4 = routing_lines4.rename(columns={0: 'geometry'}).set_geometry('geometry')
routing_lines4.crs = "EPSG:32632"
routing_network = [routing_lines, routing_lines1, routing_lines4]

map_polys = gpd.read_file("data/planEG_polygon_semantic.geojson")
map_polys1 = gpd.read_file("data/plan1OG_polygon_semantic.geojson")
map_polys4 = gpd.read_file("data/Poly-Plan4OGv3-Semantic.geojson")

map_polys.crs = "EPSG:32632"
map_polys1.crs = "EPSG:32632"
map_polys4.crs = "EPSG:32632"

all_walls = map_polys.loc[map_polys['Type'] == 'Wall']
all_walls1 = map_polys1.loc[map_polys1['Type'] == 'Wall']
# print('test',all_walls1.isna())
all_walls1 = all_walls1.dropna()

all_walls4 = map_polys4.loc[map_polys4['Type'] == 'Wall']

gallery4 = map_polys4.loc[map_polys4['Type'] == 'Gallery']

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

only_Stairs = map_polys.loc[map_polys['Type'] == 'Stairs']
only_Stairs1 = map_polys1.loc[map_polys1['Type'] == 'Stairs']
only_Stairs4 = map_polys4.loc[map_polys4['Type'] == 'Stairs']

only_Door = map_polys.loc[map_polys['Type'] == 'Door']
only_Door1 = map_polys1.loc[map_polys1['Type'] == 'Door']
only_Door4 = map_polys4.loc[map_polys4['Type'] == 'Door']

# the spatial queries are a little quicker when first creating the union of all geometries into one multipolygon:
union_walls = all_walls.geometry.unary_union
union_walls_gdf = gpd.GeoDataFrame(geometry=[union_walls])
union_walls_gdf.crs = "EPSG:32632"
# print(all_walls1)
all_walls1.is_empty
union_walls1 = all_walls1.geometry.unary_union
union_walls1_gdf = gpd.GeoDataFrame(geometry=[union_walls1])
union_walls1_gdf.crs = "EPSG:32632"

union_walls4 = all_walls4.geometry.unary_union
union_walls4_gdf = gpd.GeoDataFrame(geometry=[union_walls4])
union_walls4_gdf.crs = "EPSG:32632"

# In[8]:


walls = [union_walls_gdf, union_walls1_gdf, union_walls4_gdf]
# wallsb = [all_walls,all_walls1,all_walls]


# In[9]:


union_routing_lines = routing_lines.geometry.unary_union

union_routing_lines1 = routing_lines1.geometry.unary_union

union_routing_lines4 = routing_lines4.geometry.unary_union

union_Door = only_Door.geometry.unary_union
union_Door1 = only_Door1.geometry.unary_union
union_Door4 = only_Door4.geometry.unary_union
doors = [union_Door, union_Door1, union_Door4]

# In[10]:


# "transition areas" between floors (lifts and staircases)
transition = only_Stairscase.append(Lift).append(only_Stairs)
transition1 = only_Stairscase1.append(Lift1).append(only_Stairs1)
transition4 = only_Stairscase4.append(Lift4).append(only_Stairs4)
transitions = [transition, transition1, transition4]
# In[11]:


joined_fur_base_plot = all_walls.geometry.append(only_Door.geometry)
joined_fur_base_plot1 = all_walls1.geometry.append(only_Door1.geometry)
joined_fur_base_plot4 = all_walls4.geometry.append(only_Door4.geometry)
base_plot = [joined_fur_base_plot, joined_fur_base_plot1, joined_fur_base_plot4]

last_room_global = only_Stairscase.append(only_Corridor).append(only_Door).append(only_rooms).append(Lift)
last_room_global1 = only_Stairscase1.append(only_Corridor1).append(only_Door1).append(only_rooms1).append(Lift1)
# last_room_global4 = only_Stairscase4.append(only_Corridor4).append(only_Door4).append(only_rooms4).append(Lift4)
last_room_global4 = only_Stairscase4.append(only_Corridor4).append(only_Door4).append(only_rooms4).append(Lift4).append(
    gallery4)
global_rooms = [last_room_global, last_room_global1, last_room_global4]
