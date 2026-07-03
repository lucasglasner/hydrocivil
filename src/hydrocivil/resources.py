'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 1969-12-31 21:00:00
 Modified by: Lucas Glasner,
 Modified time: 2026-05-06 16:40:13
 Description:
 Dependencies:
'''

import os
import geopandas as gpd
import rioxarray as rxr
import xarray as xr

from typing import Tuple


def load_swampy_data() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame,
                                xr.DataArray, xr.DataArray]:
    """
    Load base example data
    Returns:
        (tuple): (basin polygon, river network segments, dem, curve_number)
    """
    root_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_folder, 'resources', 'RioGomez')
    basin_path = os.path.join(data_folder, 'Cuenca_RioGomez_v0.shp')
    rivers_path = os.path.join(data_folder, 'rnetwork_RioGomez_v0.shp')
    dem_path = os.path.join(data_folder, 'DEM_ALOSPALSAR_RioGomez_v0.tif')
    cn_path = os.path.join(data_folder, 'CurveNumber_RioGomez_v0.tif')

    basin = gpd.read_file(basin_path)
    rivers = gpd.read_file(rivers_path)
    dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
    cn = rxr.open_rasterio(cn_path, masked=True).squeeze()
    dem.name = 'elevation'
    cn.name = 'cn'
    return (basin, rivers, dem, cn)


def load_steep_data() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame,
                               xr.DataArray, xr.DataArray]:
    """
    Load base example data
    Returns:
        (tuple): (basin polygon, river network segments, dem, curve_number)
    """
    root_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_folder, 'resources', 'CNT2420_2')
    basin_path = os.path.join(data_folder, 'CN-T-2420_2.shp')
    rivers_path = os.path.join(data_folder, 'rnetwork.shp')
    dem_path = os.path.join(data_folder, 'dem.tif')
    cn_path = os.path.join(data_folder, 'cn.tif')

    basin = gpd.read_file(basin_path)
    rivers = gpd.read_file(rivers_path)
    dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
    cn = rxr.open_rasterio(cn_path, masked=True).squeeze()
    dem.name = 'elevation'
    cn.name = 'cn'
    return (basin, rivers, dem, cn)


def load_urban_data() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame,
                               xr.DataArray, xr.DataArray]:
    """
    Load base example data
    Returns:
        (tuple): (basin polygon, river network segments, dem, curve_number)
    """
    root_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_folder, 'resources', 'EsteroVDM')
    basin_path = os.path.join(data_folder, 'EsteroVDM.gpkg')
    rivers_path = os.path.join(data_folder, 'rnetwork.gpkg')
    dem_path = os.path.join(data_folder, 'dem.tif')
    cn_path = os.path.join(data_folder, 'cn.tif')

    basin = gpd.read_file(basin_path)
    rivers = gpd.read_file(rivers_path)
    dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
    cn = rxr.open_rasterio(cn_path, masked=True).squeeze()
    dem.name = 'elevation'
    cn.name = 'cn'
    return (basin, rivers, dem, cn)


def load_small_data() -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame,
                               xr.DataArray, xr.DataArray]:
    """
    Load base example data
    Returns:
        (tuple): (basin polygon, river network segments, dem, curve_number)
    """
    root_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(root_folder, 'resources', 'R5A5P85_493')
    basin_path = os.path.join(data_folder, 'basin.shp')
    rivers_path = os.path.join(data_folder, 'rnetwork.shp')
    dem_path = os.path.join(data_folder, 'dem.tif')
    cn_path = os.path.join(data_folder, 'cn.tif')

    basin = gpd.read_file(basin_path)
    rivers = gpd.read_file(rivers_path)
    dem = rxr.open_rasterio(dem_path, masked=True).squeeze()
    cn = rxr.open_rasterio(cn_path, masked=True).squeeze()
    dem.name = 'elevation'
    cn.name = 'cn'
    return (basin, rivers, dem, cn)


def load_example_data(kind: str = 'swampy') -> Tuple[gpd.GeoDataFrame,
                                                     gpd.GeoDataFrame,
                                                     xr.DataArray,
                                                     xr.DataArray]:
    """
    Load base example data
    Returns:
        (tuple): (basin polygon, river network segments, dem, curve_number)
    """
    if kind == 'swampy':
        return load_swampy_data()
    elif kind == 'steep':
        return load_steep_data()
    elif kind == 'urban':
        return load_urban_data()
    elif kind == 'small':
        return load_small_data()
    else:
        types = 'swampy', 'steep', 'urban', 'small'
        text = f"kind = {kind}, expected one of: {types}"
        raise RuntimeError(text)
