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

from typing import Any, Tuple
from osgeo import gdal, gdal_array
import networkx as nx

from .misc import gdal2xarray, xarray2gdal
from .abstractions import SCS_MaximumRetention

# ------------------------ Geomorphological properties ----------------------- #


def process_gdaldem(dem: xr.DataArray, varname: str, **kwargs: Any
                    ) -> xr.DataArray:
    """
    Processes a Digital Elevation Model (DEM) using the GDAL DEMProcessing
    utility. This method utilizes the GDAL DEMProcessing command line
    utility to derive various properties from a DEM. The output is returned
    as an xarray Dataset.

    Args:
        dem (xr.DataArray): Digital elevation model
        varname (str): The name of the DEM derived property to compute.
            Options include: 'hillshade', 'slope', 'aspect', 'color-relief',
            'TRI', 'TPI', 'Roughness'
        **kwargs (Any): Additional keyword arguments to pass to the GDAL
            DEMProcessing https://gdal.org/en/stable/programs/gdaldem.html. 

    Returns:
        xr.Dataset: An xarray Dataset containing the DEM derived property.
    """
    dem_gdal = xarray2gdal(dem)

    # Create in-memory output GDAL dataset
    dtype = gdal_array.NumericTypeCodeToGDALTypeCode(dem.dtype)
    mem_driver = gdal.GetDriverByName('MEM')
    out_ds = mem_driver.Create('', dem.sizes['x'], dem.sizes['y'], 1,
                               dtype)
    out_ds.SetGeoTransform(dem.rio.transform().to_gdal())
    out_ds.SetProjection(dem.rio.crs.to_wkt())

    # Process DEM using gdal.DEMProcessing
    out_ds = gdal.DEMProcessing(out_ds.GetDescription(), dem_gdal, varname,
                                format='MEM', computeEdges=True, **kwargs)
    out_ds = gdal2xarray(out_ds).to_dataset(name=varname)
    out_ds.coords['y'] = dem.coords['y']
    out_ds.coords['x'] = dem.coords['x']
    out_ds = out_ds.where(~dem.isnull())
    return out_ds


def rivers2graph(gdf_segments: gpd.GeoDataFrame, multigraph=False) -> nx.DiGraph:
    """
    Generate a networkx directed graph following a geodataframe with river
    segments. 

    Args:
        gdf_segments (gpd.GeoDataFrame): River segments.
        multigraph (bool, optional): Whether to build the graph as a 
        multigraph or not. A multigraph accepts multiple edges between node
        so it is suitable only for braided rivers. Defaults to False.

    Returns:
        (nx.DiGraph): River network represented as a directed acyclic graph.
    """
    # Explode geodataframe into all possible segments
    gdf_segments = gdf_segments.explode(index_parts=False)

    # Remove empty geometries and compute length
    gdf_segments = gdf_segments[~gdf_segments['geometry'].is_empty]
    gdf_segments = gdf_segments.reset_index(drop=True)
    gdf_segments['length'] = gdf_segments.geometry.length

    # Create a networkx directed acyclic graph with edge attributes
    G = nx.MultiDiGraph() if multigraph else nx.DiGraph()
    G.graph["crs"] = gdf_segments.crs

    key = 0
    for ids, row in zip(gdf_segments.index, gdf_segments.itertuples()):
        first = row.geometry.coords[0]
        last = row.geometry.coords[-1]

        data = [r for r in row][1:]
        attributes = dict(zip(gdf_segments.columns, data))
        if multigraph:
            G.add_edge(first, last, segment_id=ids, key=key, **attributes)
            key += 1
        else:
            G.add_edge(first, last, segment_id=ids, **attributes)

    # Assign a topological order
    topo_order = list(nx.topological_sort(G))
    node_topo_order = {node: i for i, node in enumerate(topo_order)}
    edge_topo_order = {(u, v): node_topo_order[u] for u, v in G.edges()}
    nx.set_edge_attributes(G, edge_topo_order, "topological_order")
    gdf_segments['topological_order'] = list(edge_topo_order.values())
    gdf_segments = gdf_segments.sort_values(by='topological_order')
    return G, gdf_segments


def get_main_river(river_network: gpd.GeoSeries | gpd.GeoDataFrame
                   ) -> gpd.GeoSeries | gpd.GeoDataFrame:
    """
    For a given river network (shapefile with river segments) this functions
    creates a graph with the river network and computes the main river with the
    longest_path algorithm. 

    Args:
        river_network (GeoDataFrame): River network (lines)
    Returns:
        (GeoDataFrame): Main river extracted from the river network
    """
    # Create River Network Graph
    G, gdf_segments = rivers2graph(river_network)

    # Get the main river segments
    mriver_nodes = nx.dag_longest_path(G, weight='length')
    mriver_edges = list(zip(mriver_nodes[:-1], mriver_nodes[1:]))
    mask = [G.edges[edge]['segment_id'] for edge in mriver_edges]
    main_river = gdf_segments.loc[mask]

    top_order = [G.edges[edge]['topological_order'] for edge in mriver_edges]
    main_river['topological_order'] = top_order
    main_river = main_river.sort_values(by='topological_order')
    return main_river


def basin_outlet(basin: gpd.GeoSeries | gpd.GeoDataFrame,
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
    dx = abs((max(dem.y.diff('y')[0], dem.x.diff('x')[0])).item())
    basin_boundary = basin.boundary
    dem_boundary = dem.rio.clip(basin_boundary.buffer(dx*n))
    dem_boundary = dem_boundary.where(dem_boundary != -9999)
    outlet_point = dem_boundary.isel(**dem_boundary.argmin(['y', 'x']))
    outlet_y, outlet_x = outlet_point.y.item(), outlet_point.x.item()
    return (outlet_y, outlet_x)


def terrain_exposure(aspect: xr.DataArray,
                     direction_ranges={'N_exposure': (337.5, 22.5),
                                       'S_exposure': (157.5, 202.5),
                                       'E_exposure': (67.5, 112.5),
                                       'W_exposure': (247.5, 292.5),
                                       'NE_exposure': (22.5, 67.5),
                                       'SE_exposure': (112.5, 157.5),
                                       'SW_exposure': (202.5, 247.5),
                                       'NW_exposure': (292.5, 337.5)},
                     **kwargs) -> pd.DataFrame:
    """
    Compute terrain exposure from an aspect raster.

    Calculates the percentage of the raster area that faces each of the 
    eight cardinal and intercardinal directions (N, S, E, W, NE, SE, SW, NW),
    based on aspect values.

    Args:
        aspect: An xarray.DataArray representing terrain aspect in degrees.
                Values should range from 0 to 360, where 0/360 = North, 
                90 = East, 180 = South, and 270 = West.
        direction_ranges: A dictionary mapping direction labels to tuples
                          defining angular ranges in degrees. Defaults to
                          standard 8-direction bins.
        **kwargs are passed to pandas.Series constructor
    """
    # Direction of exposure
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
    dir_perc = pd.Series(dir_perc.values(),
                         index=dir_perc.keys(), **kwargs)
    return dir_perc
# -------------------- Concentration time for rural basins ------------------- #


def tc_SCS(mriverlen: int | float, meanslope: int | float,
           curvenumber: int | float, **kwargs: Any) -> float:
    """
    USA Soil Conservation Service (SCS) method.
    Valid for rural basins ¿?.

    Reference:
        Part 630 National Engineering Handbook. Chapter 15. NRCS 
        United States Department of Agriculture.

    Args:
        mriverlen (float): Main river length in (km)
        meanslope (float): Mean slope in m/m
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


def tc_kirpich(mriverlen: int | float, meanslope: int | float,
               **kwargs: Any) -> float:
    """
    Kirpich equation method.
    Valid for small and rural basins ¿?.

    Reference:
        ???

    Args:
        mriverlen (float): Main river length in (km)
        meanslope (float): Mean slope in m/m
        **kwargs do nothing

    Returns:
        Tc (float): Concentration time (minutes)
    """
    Tc = 3.97*(mriverlen ^ 0.77)*(meanslope ^ -0.385)
    return Tc


def tc_giandotti(mriverlen: int | float, hmean: int | float,
                 hmin: int | float, area: int | float,
                 validate: bool = True,
                 **kwargs: Any) -> float:
    """
    Giandotti equation method.
    Valid for small basins with high slope ¿?. 

    Reference:
        Volumen 3, Manual de Carreteras 1995. Tabla 3.702.501A
        Giandotti, M., 1934. Previsione delle piene e delle magre dei corsi
            d’acqua. Istituto Poligrafico dello Stato, 8, 107–117.

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
    Tc = a/b
    if validate:
        if (Tc >= mriverlen/5.4) and (Tc <= mriverlen/3.6):
            return Tc*60
        else:
            text = "Giandotti: The condition 'L/3.6 >= Tc >= L/5.4' was not met!"
            warnings.warn(text)
            return np.nan


def tc_california(mriverlen: int | float, hmax: int | float,
                  hmin: int | float, **kwargs: Any) -> float:
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


def tc_spain(mriverlen: int | float, meanslope: int | float,
             **kwargs: Any) -> float:
    """
    Equation of Spanish/Spain regulation.

    Reference:
        ???

    Args:
        mriverlen (float): Main river length in (km)
        meanslope (float): Mean slope in m/m
        **kwargs do nothing

    Returns:
        Tc (float): Concentration time (minutes)
    """
    Tc = 18*(mriverlen**0.76)/((meanslope*100)**0.19)
    return Tc


def tc_bransbywilliams(area: int | float,
                       mriverlen: int | float,
                       meanslope: int | float) -> float:
    """
    Calculates the concentration time (Tc) using the Bransby Williams formula.

    Args:
        area (int | float): Drainage area in (km2).
        mriverlen (int | float): Main river length in (km)
        meanslope (int | float): Mean slope in m/m
    Returns:
        Tc (float): Concentration time (minutes).
    """
    Tc = 14.46 * (mriverlen) / (meanslope ** 0.2) / (area ** 0.1)
    return Tc


@np.vectorize
def concentration_time(method: str, **kwargs: Any) -> float:
    """
    General function for computing the concentration time with different
    formulas. This version supports both scalar and vectorized inputs.

    Args:
        method (str): Concentration time formula:
            Options:
                California, Giandotti, Kirpich, SCS, Spain
        **kwargs are given to the respective concentration time formula

    Raises:
        ValueError: If user asks for an unknown method

    Returns:
        (float): Concentration time (minutes)
    """
    methods = {
        'California': tc_california,
        'Giandotti': tc_giandotti,
        'Kirpich': tc_kirpich,
        'SCS': tc_SCS,
        'Spain': tc_spain,
        'BransbyWilliams': tc_bransbywilliams
    }

    if method not in methods:
        raise ValueError(f'"{method}": Unknown tc method!')

    return methods[method](**kwargs)
