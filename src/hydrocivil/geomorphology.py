'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 1969-12-31 21:00:00
 Modified by: Lucas Glasner, 
 Modified time: 2024-05-06 09:56:20
 Description:
 Dependencies:
'''

import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
import warnings

from typing import Union, Any, Tuple
from numpy.typing import ArrayLike
from shapely.geometry import Point
import networkx as nx

from .abstractions import SCS_MaximumRetention

# ------------------------ Geomorphological properties ----------------------- #


def get_main_river(river_network: Union[gpd.GeoSeries, gpd.GeoDataFrame]
                   ) -> Union[gpd.GeoSeries, gpd.GeoDataFrame]:
    """
    For a given river network (shapefile with river segments) this functions
    creates a graph with the river network and computes the main river with the
    longest_path algorithm. 

    Args:
        river_network (GeoDataFrame): River network (lines)
    Returns:
        (GeoDataFrame): Main river extracted from the river network
    """
    # Get network connectivity information
    river_network = river_network.explode(index_parts=False)
    start_node = river_network.geometry.apply(lambda g: Point(g.coords[0]))
    end_node = river_network.geometry.apply(lambda g: Point(g.coords[-1]))
    weight = river_network.length
    ids = river_network.index

    # Create River Network Graph
    G = nx.DiGraph()
    for n, a, b, w in zip(ids, start_node, end_node, weight):
        G.add_edge(a, b, index=n, weight=w)

    # Get the main river segments
    main_river = nx.dag_longest_path(G)
    mask = start_node.isin(main_river)
    main_river = river_network.loc[mask]
    return main_river


def basin_outlet(basin: Union[gpd.GeoSeries, gpd.GeoDataFrame],
                 dem: xr.DataArray, n: int = 3
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function computes the basin outlet point defined as the
    point of minimum elevation along the basin boundary.

    Args:
        basin (geopandas.GeoDataFrame): basin polygon
        dem (xarray.DataArray): Digital elevation model
        n (int, optional): Number of DEM pixels to consider for the
            elevation boundary. Defaults to 3.

    Returns:
        outlet_y, outlet_x (tuple): Tuple with defined outlet y and x
            coordinates.
    """
    dx = (max(dem.y.diff('y')[0], dem.x.diff('x')[0])).item()
    basin_boundary = basin.boundary
    dem_boundary = dem.rio.clip(basin_boundary.buffer(dx*n))
    dem_boundary = dem_boundary.where(dem_boundary != -9999)
    outlet_point = dem_boundary.isel(**dem_boundary.argmin(['y', 'x']))
    outlet_y, outlet_x = outlet_point.y.item(), outlet_point.x.item()
    return (outlet_y, outlet_x)


def basin_geographical_params(fid: Union[str, int, float],
                              basin: Union[gpd.GeoSeries, gpd.GeoDataFrame],
                              outlet: ArrayLike = None) -> pd.DataFrame:
    """
    Given a basin id and a basin polygon as a geopandas object 
    this function computes the "geographical" or vector properties of
    the basin (i.e centroid coordinates, area, perimeter and outlet to
    centroid length.)

    Args:
        fid (str|int|float): basin identifier
        basin (geopandas.GeoSeries|geopandas.GeoDataFrame): basin polygon
        outlet (ArrayLike, optional): shape (2,) array with basin x,y outlet
            point. Defaults to None
    Raises:
        RuntimeError: If outlet == None and the basin doesnt have the drainage
            point in the attribute table. (outlet_x and outlet_y columns)

    Returns:
        pandas.DataFrame: table with parameters
    """
    if type(outlet) != type(None):
        outlet_x, outlet_y = (outlet[0], outlet[1])
        basin['outlet_x'] = outlet_x
        basin['outlet_y'] = outlet_y

    if not (('outlet_x' in basin.columns) or ('outlet_y' in basin.columns)):
        error = 'Basin polygon attribute table must have an "outlet_x" and'
        error = error+' "outlet_y" columns.'
        error = error+' If not, use the outlet argument.'
        raise RuntimeError(error)

    params = pd.DataFrame([], index=[fid])
    params['outlet_x'] = basin.outlet_x.item()
    params['outlet_y'] = basin.outlet_y.item()
    params['centroid_x'] = basin.centroid.x.item()
    params['centroid_y'] = basin.centroid.y.item()
    params['area'] = basin.area.item()/1e6
    params['perim'] = basin.boundary.length.item()/1e3

    # Outlet to centroid
    outlet = Point(basin.outlet_x.item(), basin.outlet_y.item())
    out2cen = basin.centroid.distance(outlet)
    params['out2centroidlen'] = out2cen.item()/1e3

    return params


def terrain_exposure(aspect: xr.DataArray, fid: Union[str, int, float] = 0
                     ) -> pd.DataFrame:
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
        'N_exposure': (337.5, 22.5),
        'S_exposure': (157.5, 202.5),
        'E_exposure': (67.5, 112.5),
        'W_exposure': (247.5, 292.5),
        'NE_exposure': (22.5, 67.5),
        'SE_exposure': (112.5, 157.5),
        'SW_exposure': (202.5, 247.5),
        'NW_exposure': (292.5, 337.5),
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
        dir_perc[direction] = (direction_pixels/tot_pixels)
    dir_perc = pd.DataFrame(dir_perc.values(),
                            index=dir_perc.keys(),
                            columns=[fid]).T
    return dir_perc


def basin_terrain_params(fid: Union[str, int, float], dem: xr.DataArray
                         ) -> pd.DataFrame:
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
    params['hmin'] = dem.elevation.min().item()
    params['hmax'] = dem.elevation.max().item()
    params['hmean'] = dem.elevation.mean().item()
    params['hmed'] = dem.elevation.median().item()
    params['deltaH'] = params['hmax']-params['hmin']
    params['deltaHm'] = params['hmean']-params['hmin']

    # Slope parameters
    if 'slope' in dem.variables:
        params['meanslope'] = dem.slope.mean().item()
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


def tc_SCS(mriverlen: Union[int, float], meanslope: Union[int, float],
           curvenumber: Union[int, float], **kwargs: Any) -> float:
    """
    USA Soil Conservation Service (SCS) method.
    Valid for rural basins ¿?.

    Reference:
        Part 630 National Engineering Handbook. Chapter 15. NRCS 
        United States Department of Agriculture.

    Args:
        mriverlen (float): Main river length in (km)
        meanslope (float): Basin mean slope in m/m
        curvenumber (float): Basin curve number (dimensionless)
        **kwargs do nothing

    Returns:
        Tc (float): Concentration time (minutes)
    """
    mriverlen_ft = 3280.84*mriverlen
    potentialstorage_inch = SCS_MaximumRetention(curvenumber, cfactor=1)
    slope_perc = meanslope*100
    numerator = mriverlen_ft**0.8*((potentialstorage_inch+1) ** 0.7)
    denominator = 1140*slope_perc**0.5
    Tc = numerator/denominator*60  # 60 minutes = 1 hour
    return Tc


def tc_kirpich(mriverlen: Union[int, float], hmax: Union[int, float],
               hmin: Union[int, float], **kwargs: Any) -> float:
    """
    Kirpich equation method.
    Valid for small and rural basins ¿?.

    Reference:
        ???

    Args:
        mriverlen (float): Main river length in (km)
        hmax (float): Basin maximum height (m)
        hmin (float): Basin minimum height (m)
        **kwargs do nothing

    Returns:
        Tc (float): Concentration time (minutes)
    """
    deltaheights = hmax-hmin
    Tc = ((1000*mriverlen)**1.15)/(deltaheights**0.385)/51
    return Tc


def tc_giandotti(mriverlen: Union[int, float], hmean: Union[int, float],
                 hmin: Union[int, float], area: Union[int, float],
                 **kwargs: Any) -> float:
    """
    Giandotti equation method.
    Valid for small basins (< 20km2) with high slope (>10%) ¿?. 

    Reference:
        ???

    Args:
        mriverlen (float): Main river length in (km)
        hmean (float): Basin mean height (meters)
        hmin (float): Basin minimum height (meters)
        area (float): Basin area (km2)
        **kwargs do nothing

    Returns:
        Tc (float): Concentration time (minutes)
    """
    a = (4*area**0.5+1.5*mriverlen)
    b = (0.8*(hmean-hmin)**0.5)
    Tc = 60*a/b
    return Tc


def tc_california(mriverlen: Union[int, float], hmax: Union[int, float],
                  hmin: Union[int, float], **kwargs: Any) -> float:
    """
    California Culverts Practice (1942) equation.
    Valid for mountain basins ¿?.

    Reference: 
        ???

    Args:
        mriverlen (float): Main river length in (km)
        hmax (float): Basin maximum height (m)
        hmin (float): Basin minimum height (m)
        **kwargs do nothing

    Returns:
        Tc (float): Concentration time (minutes)

    """
    deltaheights = hmax-hmin
    Tc = 57*(mriverlen**3/deltaheights)**0.385
    return Tc


def tc_spain(mriverlen: Union[int, float], meanslope: Union[int, float],
             **kwargs: Any) -> float:
    """
    Equation of Spanish/Spain regulation.

    Reference:
        ???

    Args:
        mriverlen (float): Main river length in (km)
        meanslope (float): Basin mean slope in m/m
        **kwargs do nothing

    Returns:
        Tc (float): Concentration time (minutes)
    """
    Tc = 18*(mriverlen**0.76)/((meanslope*100)**0.19)
    return Tc


def concentration_time(method: str, **kwargs: Any) -> float:
    """
    General function for computing the concentration time with different
    formulas.

    Args:
        method (str): Concentration time formula:
            Options:
                California, Giandotti, Kirpich, SCS, Spain, 
        **kwargs are given to the respective concentration time formula

    Raises:
        ValueError: If user asks for an unkown method

    Returns:
        (float): Concentration time (minutes)
    """
    if method == 'California':
        return tc_california(**kwargs)
    elif method == 'Giandotti':
        return tc_giandotti(**kwargs)
    elif method == 'Kirpich':
        return tc_kirpich(**kwargs)
    elif method == 'SCS':
        return tc_SCS(**kwargs)
    elif method == 'Spain':
        return tc_spain(**kwargs)
    else:
        raise ValueError(f'"{method}": Unknown tc method!')
