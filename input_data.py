import geopandas as gpd
import numpy as np
import shapely
import shapely.wkt
from shapely.geometry import *
# gpd.options.use_pygeos = True
from mathematics import azimuthAngle, calc_azimuth
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename, askdirectory

def adjust_freqByRate(fg, rate):
    new_fg = []

    for i, g in enumerate(fg):
        if i == 0:
            new_fg.append(g)
        elif g[0] - new_fg[-1][0] > rate * 1000:
            new_fg.append(g)
    return np.array(new_fg)

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


def Load_Step_Data(_trajectory,rate):
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
    '''if _trajectory == 'test':
        # for eight path:
        SHeading = np.genfromtxt('data/eight/StepHeadigs.csv', delimiter=',', skip_header=0)
        SLengthraw = np.genfromtxt('data/eight/StepLengths.csv', delimiter=',', skip_header=0)
        SLength = SLengthraw
        ref = np.genfromtxt('data/eight/ref.csv', delimiter=' ', skip_header=0)
        SHeight = np.genfromtxt('data/eight/DeltaHeight.csv', delimiter=' ', skip_header=0)'''


    if _trajectory == 'kf':
        # for eight path:
        print('select positions(steps)!')
        position_file = '/Users/hdr026/Nextcloud/ION_and_IVK/optimisation_data/wls_data/0509_take02/wls_rate2s.csv'#'C:/Users/dmz-admin/Nextcloud/ION_and_IVK/optimisation_data/wls_data/0509_take02/wls_rate2s.csv'#'/Users/hdr026/Nextcloud/ION_and_IVK/optimisation_data/wls_data/0509_take02/wls_rate2s.csv'#askopenfilename()
        kf_positions = np.genfromtxt(position_file, delimiter=',', skip_header=1)
        kf_points_gdf = gpd.GeoSeries([Point(k[1],k[2]) for k in kf_positions])
        delta_x = []
        delta_y = []

        SLength = []
        SHeading = []
        step_time = [kf_positions[0][0]]
        for i in range(len(kf_positions)-1):
            dx = kf_positions[i+1][1] - kf_positions[i][1]
            dy = kf_positions[i+1][2] - kf_positions[i][2]
            delta_x.append(dx)
            delta_y.append(dy)
            step_time.append(kf_positions[i+1][0])

            SL = np.sqrt(dx**2 + dy**2)
            SH = calc_azimuth([kf_positions[i][1],kf_positions[i][2]], [kf_positions[i+1][1],kf_positions[i+1][2]])
            SHeading.append(SH)
            SLength.append(SL)
            #SHeading.append(azimuthAngle(kf_positions[i][1], kf_positions[i][2], kf_positions[i+1][1], kf_positions[i+1][2]))

        print('select reference!')
        ref_file = '/Users/hdr026/Nextcloud/IIS_Campaign/campaign_09_2022/IIS_daten/05092022/2/Sync/synced_data_nps_rate_5.csv'#'C:/Users/dmz-admin/Nextcloud/IIS_Campaign/campaign_09_2022/IIS_daten/05092022/2/Sync/synced_data_nps_rate_5.csv'#'/Users/hdr026/Nextcloud/IIS_Campaign/campaign_09_2022/IIS_daten/05092022/2/Sync/synced_data_nps_rate_5.csv'#askopenfilename()
        ref = np.genfromtxt(ref_file, delimiter=',', skip_header=1)
        SHeight = [0 for i in range(len(SLength))]


    ###reference points from path:
    ref_list = []
    for r in ref:
        if np.isnan(r[8]) == False:
            ref_list.append(Point(r[8], r[9]))

    gs_ref = gpd.GeoSeries(ref_list)

    ref_clean = [[r[1], r[8], r[9]] for r in ref if np.isnan(r[8]) == False]
    g5_list = [[r[1],r[2],r[3]] for r in ref]
    g5_list = adjust_freqByRate(g5_list, rate)
    
    
    return SHeading, SLength, ref_clean, SHeight, gs_ref,kf_points_gdf,g5_list,step_time


'''routing_lines = gpd.read_file("data/routeEG.geojson")
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
map_polys4.crs = "EPSG:32632"'''

all_walls = gpd.read_file("data/6_walls_a_no_door.geojson")
inner_walls = gpd.read_file("data/4_walls_a_no_door.geojson")
# print('test',all_walls1.isna())
#all_walls = all_walls.dropna()


'''only_Door = map_polys.loc[map_polys['Type'] == 'Door']
only_Door1 = map_polys1.loc[map_polys1['Type'] == 'Door']
only_Door4 = map_polys4.loc[map_polys4['Type'] == 'Door']
'''
# the spatial queries are a little quicker when first creating the union of all geometries into one multipolygon:
union_walls = all_walls.geometry.unary_union
union_walls_gdf = gpd.GeoDataFrame(geometry=[union_walls])
union_walls_gdf.crs = "EPSG:32632"
# print(all_walls1)


# In[8]:


walls = [all_walls]




joined_fur_base_plot = inner_walls.geometry
base_plot = [joined_fur_base_plot, joined_fur_base_plot, joined_fur_base_plot]
