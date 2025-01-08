'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 1969-12-31 21:00:00
 Modified by: Lucas Glasner, 
 Modified time: 2024-05-06 16:59:31
 Description:
 Dependencies:
'''

import os
import pandas as pd
import numpy as np

import geopandas as gpd
import rioxarray as rxr
import xarray as xr

from osgeo import gdal, gdal_array
from rasterio.features import shapes
from shapely.geometry import shape

from typing import Union, Tuple, Any
from numpy.typing import ArrayLike
from .global_vars import GDAL_EXCEPTIONS

if GDAL_EXCEPTIONS:
    gdal.UseExceptions()
else:
    gdal.DontUseExceptions()

# ------------------------------------ gis ----------------------------------- #


def polygonize(da: xr.DataArray, filter_areas: float = 0) -> gpd.GeoDataFrame:
    """
    Polygonize a boolean rioxarray raster
    Args:
        da (xarray.DataArray): Loaded raster as an xarray object. This should
            have typical rioxarray attributes like da.rio.crs
            and da.rio.transform
        filter_areas (float, optional): Remove polygons with an area less than
            filter areas. Defaults to 0.

    Returns:
        (geopandas.GeoDataFrame): polygonized boolean raster
    """
    da = da.astype(int)
    polygons = []
    for s, v in shapes(da, transform=da.rio.transform()):
        if v == 1:
            polygons.append(shape(s))
    polygons = gpd.GeoSeries(polygons)
    polygons = gpd.GeoDataFrame(geometry=polygons, crs=da.rio.crs)
    polygons = polygons.where(polygons.area > filter_areas).dropna()
    return polygons


def obj_to_xarray(obj: ArrayLike,
                  **kwargs: Any) -> Union[xr.DataArray, xr.Dataset]:
    """
    This function recieves an object and transform it to an xarray object.
    Only accepts pandas series, dataframes, numpy arrays, lists, tuples, or
    array like objects.

    Args:
        obj (ArrayLike): input n-dimensional array.
        **kwargs (optional): Additional keyword arguments to pass to
            xarray.DataArray constructor

    Raises:
        RuntimeError: If input object is not a pandas series, dataframe, 
            list, tuple, int, float, xarray or numpy object.

    Returns:
        (xarray): Transformed array into an xarray style.
    """
    if isinstance(obj, np.ndarray):
        new_xarray = xr.DataArray(obj, **kwargs)
    elif isinstance(obj, pd.Series) or isinstance(obj, pd.DataFrame):
        return obj_to_xarray(obj.values, **kwargs)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return obj_to_xarray(np.array(obj), **kwargs)
    elif isinstance(obj, int) or isinstance(obj, float):
        return obj_to_xarray(np.array([obj]), **kwargs)
    elif isinstance(obj, xr.DataArray) or isinstance(obj, xr.Dataset):
        new_xarray = obj
    else:
        text = f'Only ints, floats, lists, tuples, pandas, numpy or'
        text = text + f'xarray objects can be used. Got {type(obj)} instead.'
        raise RuntimeError(text)
    return new_xarray


def xarray2gdal(da: xr.DataArray, nodata: Union[str, int, float] = None
                ) -> gdal.Dataset:
    """
    Convert an xarray dataset to a GDAL in-memory dataset.

    Args:
        da (xarray.DataArray): Input data array.
        nodata (_type_, optional): Input dtype. Defaults to None.

    Returns:
        (osgeo.gdal.Dataset): GDAL in-memory dataset
    """
    # Get the data and metadata
    data = da.values
    meta = da.rio.crs.to_wkt()
    transform = da.rio.transform()
    if nodata is None:
        nodata = da.rio.nodata
    dtype = gdal_array.NumericTypeCodeToGDALTypeCode(data.dtype)

    # Write the data to a GDAL in-memory dataset
    driver = gdal.GetDriverByName('MEM')
    outRaster = driver.Create('', da.sizes['x'], da.sizes['y'], 1, dtype)

    # Set geotransform, projection and inner data
    outRaster.SetGeoTransform(transform.to_gdal())
    outRaster.SetProjection(meta)
    outRaster.GetRasterBand(1).WriteArray(data)
    return outRaster


def get_coordinates_from_gdal(gdal_ds: gdal.Dataset
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get x, y coordinates from a GDAL in-memory dataset.

    Args:
        gdal_ds (osgeo.gdal.Dataset): GDAL in-memory dataset.

    Returns:
        (np.ndarray, np.ndarray): Tuple with arrays of x,y coordinates.
    """
    # Get geotransform
    transform = gdal_ds.GetGeoTransform()

    # Get raster size
    cols = gdal_ds.RasterXSize
    rows = gdal_ds.RasterYSize

    # Calculate x coordinates (for each column)
    x_coords = np.array([transform[0] + j * transform[1] for j in range(cols)])

    # Calculate y coordinates (for each row)
    y_coords = np.array([transform[3] + i * transform[5] for i in range(rows)])

    return x_coords, y_coords


def gdal2xarray(gdal_ds: gdal.Dataset) -> xr.DataArray:
    """
    Convert a GDAL in-memory dataset to an xarray dataset.

    Args:
        gdal_ds (osgeo.gdal.Dataset): Input GDAL in-memory dataset.

    Returns:
        (xarray.DataArray): Output xarray data array.
    """
    # Get data and metadata
    data = gdal_ds.GetRasterBand(1).ReadAsArray()
    transform = rxr.raster_dataset.Affine(*gdal_ds.GetGeoTransform())
    crs = rxr.crs.CRS(gdal_ds.GetProjection())
    nodata = gdal_ds.GetRasterBand(1).GetNoDataValue()

    # Get raster coordinates
    x, y = get_coordinates_from_gdal(gdal_ds)

    # Write the data to an xarray dataarray
    da = xr.DataArray(data, dims=['y', 'x'], coords={'y': y, 'x': x})
    da = da.rio.set_spatial_dims(x_dim='x', y_dim='y')
    da = da.rio.write_crs(crs).rio.write_transform(transform)
    da.rio.update_attrs({'_FillValue': nodata})
    return da


def chile_region(point: gpd.GeoSeries, regions: gpd.GeoDataFrame) -> Any:
    """
    Given a geopandas point and region/states polygones
    this function returns the ID of the polygon where the point belongs
    (as long as an ID column exists in the region polygons)
    If an ID column doesnt exist in the region polygons this function 
    will return the polygon index.

    Args:
        point (geopandas.Series): point 
        regions (geopandas.GeoDataFrame): collection of polygons that
            represents regions or states.

    Returns:
        (Any): ID of the polygon that contains the point
    """
    if 'ID' not in regions.columns:
        regions['ID'] = regions.index

    point = point.to_crs(regions.crs)
    mask = []
    for i in range(len(regions)):
        geom = regions.geometry.iloc[i]
        mask.append(point.within(geom))
    region = regions.iloc[mask, :]['ID']
    return region.item()


def raster_distribution(raster: xr.DataArray, **kwargs: Any) -> pd.Series:
    """
    Given a raster this function computes the histogram

    Args:
        raster (xarray.DataArray): raster

    Returns:
        (pandas.Series): Histogram with data in index and pdf in values
    """
    total_pixels = raster.size
    total_pixels = total_pixels-np.isnan(raster).sum()
    total_pixels = total_pixels.item()

    values = raster.squeeze().values.flatten()
    values = values[~np.isnan(values)]
    dist, values = np.histogram(values, **kwargs)
    dist, values = dist/total_pixels, 0.5*(values[:-1]+values[1:])
    return pd.Series(dist, index=values, name=f'{raster.name}_dist')

# ----------------------------- resources access ----------------------------- #


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


def load_example_data(kind: str = 'swampy') -> Tuple[gpd.GeoDataFrame,
                                                     gpd.GeoDataFrame,
                                                     xr.DataArray,
                                                     xr.DataArray]:
    """
    Load base example data
    Returns:
        (tuple): (basin polygon, river network segments, dem, curve_number)
    """
    try:
        import geopandas as gpd
        import rioxarray as rxr
    except Exception as e:
        raise RuntimeError(e)

    if kind == 'swampy':
        return load_swampy_data()
    elif kind == 'steep':
        return load_steep_data()
    elif kind == 'urban':
        return load_urban_data()
    else:
        text = f"kind = {kind}, expected one of: 'swampy', 'steep', 'urban'"
        raise RuntimeError(text)

# ----------------------------------- other ---------------------------------- #


def get_psep() -> str:
    """
    Return path separator for the current os

    Returns:
        (str): path separator
    """
    if os.name == 'nt':
        psep = '\\'
    else:
        psep = '/'
    return psep


def is_iterable(obj: Any) -> bool:
    """
    Simple function to check if an object
    is an iterable.

    Args:
        obj (Any): Any python class

    Returns:
        (bool): True if iterable False if not
    """
    try:
        iter(obj)
        return True
    except:
        return False


def to_numeric(obj: Any) -> Any:
    """
    Simple function to transform an object to a numeric type if possible.

    Args:
        obj (Any): The object to be converted to a numeric type.

    Returns:
        (Any): The converted numeric object if conversion is possible,
            otherwise the original object.
    """
    try:
        return pd.to_numeric(obj)
    except:
        return obj
