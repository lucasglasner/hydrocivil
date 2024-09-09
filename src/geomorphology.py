'''
 # @ Author: Lucas Glasner (lgvivanco96@gmail.com)
 # @ Create Time: 1969-12-31 21:00:00
 # @ Modified by: Lucas Glasner, 
 # @ Modified time: 2024-05-06 09:56:20
 # @ Description:
 # @ Dependencies:
 '''

import pandas as pd
import numpy as np
import warnings
from shapely.geometry import Point
from src.infiltration import SCS_MaximumRetention

# ------------------------ Geomorphological properties ----------------------- #


def main_river(river_network,
               node_a='NODE_A', node_b='NODE_B',
               weight='LENGTH'):
    """
    For a given river network (shapefile with segments and connectivity
    information) this functions creates a graph with the river network
    and computes the main river with the longest_path algorithm. 

    It is recommended to use a river network computed with SAGA-GIS for a 
    straight run. 

    Args:
        river_network (GeoDataFrame): River network (lines)
        node_a (str): Column name with the starting ID of each segment
            Defalts to 'NODE_A' (Attribute by SAGA-GIS)
        node_b (str): Column name with the ending ID of each segment
            Defalts to 'NODE_B' (Attribute by SAGA-GIS)
        weight (str): Column name with segment weight
            Defalts to 'LENGTH' (Attribute by SAGA-GIS)
    Returns:
        (GeoDataFrame): Main river extracted from the river network
    """
    import networkx as nx
    # Create River Network Graph
    try:
        G = nx.DiGraph()
        for a, b, leng in zip(river_network[node_a],
                              river_network[node_b],
                              river_network[weight]):
            G.add_edge(a, b, weight=leng)

        # Get the main river segments
        main_river = nx.dag_longest_path(G)
        mask = river_network[node_a].map(lambda s: s in main_river)
        main_river = river_network.loc[mask]
        return main_river
    except Exception as e:
        warnings.warn('Couldnt compute main river:', e)
        return []


def basin_geographical_params(fid, basin):
    """
    Given a basin id and a basin polygon as a geopandas object 
    this function computes the "geographical" or vector properties of
    the basin (i.e centroid coordinates, area, perimeter and outlet to
    centroid length.)

    Args:
        fid (_type_): basin identifier
        basin (geopandas.GeoDataFrame): basin polygon

    Raises:
        RuntimeError: If the basin doesnt have the drainage point in the
            attribute table. (outlet_x and outlet_y columns)

    Returns:
        pandas.DataFrame: table with parameters
    """
    if not (('outlet_x' in basin.columns) or ('outlet_y' in basin.columns)):
        error = 'Basin polyugon attribute table must have'
        error = error+' an "outlet_x" and "outlet_y" columns'
        raise RuntimeError(error)

    params = pd.DataFrame([], index=[fid])
    params['outlet_x'] = basin.outlet_x.item()
    params['outlet_y'] = basin.outlet_y.item()
    params['centroid_x'] = basin.centroid.x.item()
    params['centroid_y'] = basin.centroid.y.item()
    params['area_km2'] = basin.area.item()/1e6
    params['perim_km'] = basin.boundary.length.item()/1e3

    # Outlet to centroid
    outlet = Point(basin.outlet_x.item(),
                   basin.outlet_y.item())
    out2cen = basin.centroid.distance(outlet)
    params['out2centroidlen_km'] = out2cen.item()/1e3

    return params


def terrain_exposure(aspect, fid=0):
    """
    From an aspect raster compute the percentage of the raster that
    belong to each of the 8 typical geographical directions.
    (i.e N, S, E, W, NE, SE, SW, NW).

    Args:
        aspect (xarray.DataArray): Aspect raster
        fid (_type_, optional): Feature ID. Defaults to 0.'

    Returns:
        pandas.DataFrame: Table with main directions exposure
    """
    # Direction of exposure
    direction_ranges = {
        'N_exposure_%': (337.5, 22.5),
        'S_exposure_%': (157.5, 202.5),
        'E_exposure_%': (67.5, 112.5),
        'W_exposure_%': (247.5, 292.5),
        'NE_exposure_%': (22.5, 67.5),
        'SE_exposure_%': (112.5, 157.5),
        'SW_exposure_%': (202.5, 247.5),
        'NW_exposure_%': (292.5, 337.5),
    }
    # Calculate percentages for each direction
    tot_pixels = np.size(aspect.values) - \
        np.isnan(aspect.values).sum()
    dir_perc = {}

    for direction, (min_angle, max_angle) in direction_ranges.items():
        if min_angle > max_angle:
            exposure = np.logical_or(
                (aspect.values >= min_angle) & (
                    aspect.values <= 360),
                (aspect.values >= 0) & (aspect.values <= max_angle)
            )
        else:
            exposure = (aspect.values >= min_angle) & (
                aspect.values <= max_angle)

        direction_pixels = np.sum(exposure)
        dir_perc[direction] = (direction_pixels/tot_pixels)*100
    dir_perc = pd.DataFrame(dir_perc.values(),
                            index=dir_perc.keys(),
                            columns=[fid]).T
    return dir_perc


def basin_terrain_params(fid, dem):
    """
    From an identifier (fid) and a digital elevation model (DEM) loaded
    as an xarray object, this function computes the following properties:
    1) Minimum, mean, median and maximum height
    2) Difference between maximum and minimum height
    3) Difference between mean and minimum height
    4) Mean slope if slope is in the dataset
    5) % of the terrain in each of the 8 directions (N,S,W,E,SW,SE,NW,NE)

    Args:
        fid (_type_): basin identifier
        dem (xarray.Dataset): Digital elevation model

    Returns:
        pandas.DataFrame: Table with terrain-derived parameters
    """
    if 'elevation' not in dem.variables:
        text = 'Input dem must be an xarray dataset with an "elevation" \
                variable'
        raise RuntimeError(text)
    params = pd.DataFrame([], index=[fid])

    # Height parameters
    params['hmin_m'] = dem.elevation.min().item()
    params['hmax_m'] = dem.elevation.max().item()
    params['hmean_m'] = dem.elevation.mean().item()
    params['hmed_m'] = dem.elevation.median().item()
    params['deltaH_m'] = params['hmax_m']-params['hmin_m']
    params['deltaHm_m'] = params['hmean_m']-params['hmin_m']

    # Slope parameters
    if 'slope' in dem.variables:
        params['meanslope_1'] = dem.slope.mean().item()
    else:
        warnings.warn('"slope" variable doesnt exists in the dataset!')

    # Exposure/Aspect parameters
    if 'aspect' in dem.variables:
        dir_perc = terrain_exposure(dem.aspect, fid=fid)
        params = pd.concat([params, dir_perc], axis=1)
    else:
        warnings.warn('"aspect" variable doesnt exists in the dataset!')
    return params

# -------------------- Concentration time for rural basins ------------------- #


def tc_SCS(basin_mriverlen_km,
           basin_meanslope_1,
           curvenumber_1):
    """
    USA Soil Conservation Service (SCS) method.
    Valid for rural basins ¿?.

    Reference:
        Part 630 National Engineering Handbook. Chapter 15. NRCS 
        Unitaded States Deprtment of Agriculture.

    Args:
        basin_mriverlen_km (float): Main river length in (km)
        basin_meanslope_1 (float): Basin mean slope in m/m
        curvenumber_1 (float): Basin curve number (dimensionless)

    Returns:
        Tc (float): Concentration time (minutes)
    """
    mriverlen_ft = 3280.84*basin_mriverlen_km
    potentialstorage_inch = SCS_MaximumRetention(curvenumber_1, cfactor=1)
    slope_perc = basin_meanslope_1*100
    numerator = mriverlen_ft**0.8*((potentialstorage_inch+1) ** 0.7)
    denominator = 1140*slope_perc**0.5
    Tc = numerator/denominator*60  # 60 minutes = 1 hour
    return Tc


def tc_kirpich(basin_mriverlen_km,
               basin_hmax_m,
               basin_hmin_m):
    """
    Kirpich equation method.
    Valid for small and rural basins ¿?.

    Reference:
        ???

    Args:
        basin_mriverlen_km (float): Main river length in (km)
        basin_hmax_m (float): Basin maximum height (m)
        basin_hmin_m (float): Basin minimum height (m)

    Returns:
        Tc (float): Concentration time (minutes)
    """
    basin_deltaheights_m = basin_hmax_m-basin_hmin_m
    Tc = ((1000*basin_mriverlen_km)**1.15)/(basin_deltaheights_m**0.385)/51
    return Tc


def tc_giandotti(basin_mriverlen_km,
                 basin_hmean_m,
                 basin_hmin_m,
                 basin_area_km2):
    """
    Giandotti equation method.
    Valid for small basins (< 20km2) with high slope (>10%) ¿?. 

    Reference:
        ???

    Args:
        basin_mriverlen_km (float): Main river length in (km)
        basin_hmean_m (float): Basin mean height (meters)
        basin_hmin_m (float): Basin minimum height (meters)
        basin_area_km2 (float): Basin area (km2)

    Returns:
        Tc (float): Concentration time (minutes)
    """
    a = (4*basin_area_km2**0.5+1.5*basin_mriverlen_km)
    b = (0.8*(basin_hmean_m-basin_hmin_m)**0.5)
    Tc = 60*a/b
    return Tc


def tc_california(basin_mriverlen_km,
                  basin_hmax_m,
                  basin_hmin_m):
    """
    California Culverts Practice (1942) equation.
    Valid for mountain basins ¿?.

    Reference: 
        ???

    Args:
        basin_mriverlen_km (float): Main river length in (km)
        basin_hmax_m (float): Basin maximum height (m)
        basin_hmin_m (float): Basin minimum height (m)

    Returns:
        Tc (float): Concentration time (minutes)

    """
    basin_deltaheights_m = basin_hmax_m-basin_hmin_m
    Tc = 57*(basin_mriverlen_km**3/basin_deltaheights_m)**0.385
    return Tc


def tc_spain(basin_mriverlen_km,
             basin_meanslope_1):
    """
    Equation of Spanish/Spain regulation.

    Reference:
        ???

    Args:
        basin_mriverlen_km (float): Main river length in (km)
        basin_meanslope_1 (float): Basin mean slope in m/m

    Returns:
        Tc (float): Concentration time (minutes)
    """
    Tc = 18*(basin_mriverlen_km**0.76)/((basin_meanslope_1*100)**0.19)
    return Tc
