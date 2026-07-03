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
import warnings

from osgeo import gdal, gdal_array
from rasterio.features import shapes
from rasterio.features import rasterize as riorasterize
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely import get_coordinates
from scipy.interpolate import interp1d

from typing import Tuple, Any
from numpy.typing import ArrayLike
from .global_vars import GDAL_EXCEPTIONS

if GDAL_EXCEPTIONS:
    gdal.UseExceptions()
else:
    gdal.DontUseExceptions()

# ------------------------------------ gis ----------------------------------- #


def minradius_from_centroid(polygon: gpd.GeoDataFrame):
    """
    Return the minimum radius of a circle centered at the polygon centroid
    that completely encloses the polygon.

    The radius is the maximum distance from the centroid to any polygon
    vertex. Works for both Polygon and MultiPolygon geometries.

    Parameters
    ----------
    polygon : GeoDataFrame, GeoSeries, Polygon or MultiPolygon
        Input geometry. If a GeoDataFrame contains multiple rows, they are
        dissolved into a single geometry.

    Returns
    -------
    radius : float
        Radius in the CRS units.

    centroid : shapely.geometry.Point
        Polygon centroid.
    """

    # Extract geometry
    if isinstance(polygon, gpd.GeoDataFrame):
        if polygon.crs is not None and polygon.crs.is_geographic:
            raise ValueError(
                "CRS is geographic. Reproject to a projected CRS first."
            )
        geom = polygon.dissolve().geometry.iloc[0]

    elif isinstance(polygon, gpd.GeoSeries):
        if polygon.crs is not None and polygon.crs.is_geographic:
            raise ValueError(
                "CRS is geographic. Reproject to a projected CRS first."
            )
        geom = polygon.unary_union

    elif isinstance(polygon, (Polygon, MultiPolygon)):
        geom = polygon

    else:
        raise TypeError(
            "polygon must be a GeoDataFrame, GeoSeries, Polygon or MultiPolygon."
        )

    centroid = geom.centroid

    # Coordinates of every polygon vertex (exterior + holes + multipolygons)
    coords = get_coordinates(geom)

    dx = coords[:, 0] - centroid.x
    dy = coords[:, 1] - centroid.y

    radius = np.hypot(dx, dy).max()

    return float(radius), centroid


def raster_cross_section(raster: xr.DataArray, line: gpd.GeoSeries,
                         **kwargs: Any) -> Tuple[pd.Series, gpd.GeoSeries]:
    """
    Sample values from a raster at specified points along a line
    (or multiple lines) and return the sampled data along with the points.

    Parameters:
    - raster (xr.DataArray): A 2D xarray containing the raster data
        (with dimensions 'x' and 'y').
    - line (gpd.GeoSeries): A GeoSeries containing the geometries of lines
        (LineString or MultiLineString) for sampling locations.
    - **kwargs (Any): Additional keyword arguments passed to the interpolation
        method for raster sampling.

    Returns:
    - Tuple[pd.Series, gpd.GeoSeries]: A tuple where:
        - The first element is a pandas Series containing the interpolated
          values at each point along the line(s).
        - The second element is a GeoSeries of Points representing the
          locations where the raster was sampled.
    """
    # Get coordinates of line points
    line_coords = []
    for segment in line.geometry:
        if segment.geom_type == "MultiLineString":
            for single_line in segment.geoms:
                line_coords.extend(list(single_line.coords))
        else:  # If it's already a LineString
            line_coords.extend(list(segment.coords))

    # Get point and x,y coordinates
    points = [Point(p[0], p[1]) for p in line_coords]
    dist = [points[i].distance(points[i+1]) for i in range(len(points)-1)]
    dist = np.hstack((0, np.cumsum(dist)))
    if dist[-1] - line.length.sum() > 1e-5:
        raise ValueError("Line length and cumulative distance do not match. "
                         "Check the line geometry.")
    xs, ys = zip(*[(p.x, p.y) for p in points])

    # Sample raster with points with interpolation
    xs = xr.DataArray(list(xs), dims=('points'))
    ys = xr.DataArray(list(ys), dims=('points'))
    data = raster.interp(x=xs, y=ys, **kwargs)
    data.coords['points'] = range(len(points))
    data.coords['dist'] = ('points', dist)
    data = data.swap_dims({'points': 'dist'})
    return data


def raster_counts(raster: xr.DataArray,
                  output_type: int = 1) -> pd.DataFrame:
    """
    Computes area distributions of rasters (% of the basin area with the
    X raster property)
    Args:
        raster (xarray.DataArray): Raster with basin properties
            (e.g land cover classes, soil types, etc)
        output_type (int, optional): Output type:
            Option 1:
                Returns a table with this format:
                +-------+----------+----------+
                | INDEX | PROPERTY | FRACTION |
                +-------+----------+----------+
                |     0 | A        |          |
                |     1 | B        |          |
                |     2 | C        |          |
                +-------+----------+----------+

            Option 2:
                Returns a table with this format:
                +-------------+----------+
                |    INDEX    | FRACTION |
                +-------------+----------+
                | fPROPERTY_A |          |
                | fPROPERTY_B |          |
                | fPROPERTY_C |          |
                +-------------+----------+

            Defaults to 1.
    Returns:
        counts (pandas.DataFrame): Results table
    """
    try:
        counts = raster.to_series().value_counts()
        counts = counts/counts.sum()
        if output_type == 1:
            counts = counts.reset_index().rename({raster.name: 'class'},
                                                 axis=1)
        elif output_type == 2:
            counts.index = [f'f{raster.name}_{i}' for i in counts.index]
            counts = pd.DataFrame(counts)
        else:
            raise RuntimeError(f'{output_type} must only be 1 or 2.')
    except Exception as e:
        if raster.name is None:
            raster.name = 'raster'
        counts = pd.DataFrame([], columns=[raster.name],
                              index=[0])
        warnings.warn('Raster counting Error:'+f'{e}')
    return counts


def rasterize(vector: gpd.GeoDataFrame, raster: xr.DataArray, **kwargs
              ) -> xr.DataArray:
    """
    Rasterize a vector layer.

    Args:
        vector (GeoDataFrame): The vector layer to rasterize.
        raster (xarray.DataArray): A reference raster to define the output
            shape and transform.

    Returns:
        (xarray.DataArray): Rasterized vector layer as a boolean xarray.
    """
    mask_array = riorasterize(
        [(geom, 1) for geom in vector.geometry],
        out_shape=raster.shape,
        transform=raster.rio.transform(),
        fill=0,
        dtype='uint8',
        **kwargs
    )
    mask_xarray = xr.DataArray(mask_array,
                               coords=raster.coords,
                               dims=raster.dims, name='mask')
    mask_xarray = mask_xarray.rio.write_crs(raster.rio.crs)
    mask_xarray = mask_xarray.astype(bool)
    return mask_xarray


def polygonize(da: xr.DataArray, filter_areas: float = 0) -> gpd.GeoDataFrame:
    """
    Polygonize a boolean rioxarray raster
    Args:
        da (xarray.DataArray): Loaded raster as an xarray object. This should
            have typical rioxarray attributes like da.rio.crs
            and da.rio.transform
        filter_areas (float, optional): Remove polygons with an area less than
            the given value. Defaults to 0.

    Returns:
        (geopandas.GeoDataFrame): polygonized boolean raster
    """
    da = da.astype("int32")
    polygons = []
    for s, v in shapes(da, transform=da.rio.transform()):
        if v == 1:
            polygons.append(shape(s))
    polygons = gpd.GeoSeries(polygons)
    polygons = gpd.GeoDataFrame(geometry=polygons, crs=da.rio.crs)

    if filter_areas > 0:
        polygons = polygons.where(polygons.area > filter_areas).dropna()

    return polygons


def obj2xarray(obj: ArrayLike, **kwargs: Any) -> xr.DataArray | xr.Dataset:
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
    elif isinstance(obj, pd.Series):
        return obj.to_xarray()
    elif isinstance(obj, pd.DataFrame):
        return obj.stack().to_xarray()
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return obj2xarray(np.array(obj), **kwargs)
    elif isinstance(obj, int) or isinstance(obj, float):
        return obj2xarray(np.array([obj]), **kwargs)
    elif isinstance(obj, xr.DataArray) or isinstance(obj, xr.Dataset):
        new_xarray = obj
    else:
        text = 'Only ints, floats, lists, tuples, pandas, numpy or'
        text = text + f'xarray objects can be used. Got {type(obj)} instead.'
        raise RuntimeError(text)
    return new_xarray.squeeze()


def xarray2gdal(da: xr.DataArray) -> gdal.Dataset:
    """
    Convert an xarray dataset to a GDAL in-memory dataset.

    Args:
        da (xarray.DataArray): Input data array.

    Returns:
        (osgeo.gdal.Dataset): GDAL in-memory dataset
    """
    # Get the data and metadata
    data = da.values
    meta = da.rio.crs.to_wkt()
    transform = da.rio.transform()
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
    crs = gdal_ds.GetProjection()
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


def sharegrids(ds1, ds2, dimnames={'x': 'x', 'y': 'y'}):
    """
    Check if two rasters have the same grid.

    Parameters
    ----------
    ds1, ds2 : xarray.DataArray
        Raster arrays to compare.
    dimnames : dict, optional
        Dictionary mapping 'x' and 'y' to coordinate names.

    Returns
    -------
    bool
        True if both rasters have the same shape and coordinates, else False.
    """
    if ds1.shape != ds2.shape:
        return False

    x1, y1 = ds1[dimnames['x']], ds1[dimnames['y']]
    x2, y2 = ds2[dimnames['x']], ds2[dimnames['y']]
    if (x1 != x2).sum() > 0:
        return False
    if (y1 != y2).sum() > 0:
        return False
    return True


# ----------------------------------- other ---------------------------------- #


def alternating_block_sort(arr: ArrayLike):
    """
    Sorts a 1D array in an alternating block pattern.

    Args:
        arr (ArrayLike): Input 1D array to be rearranged.

    Returns:
        rearranged (ArrayLike): Array with values rearranged in alternating
            block pattern, with the highest values placed near the center.
    """
    arr = np.asarray(arr).flatten()  # Ensure it's 1D
    nan_mask = np.isnan(arr)
    arr = arr[~nan_mask]

    # Sort in descending order
    arr_sorted = np.sort(arr)[::-1]

    n = arr_sorted.size
    rearranged = np.zeros_like(arr_sorted)

    # Fill the rearranged array with alternating block pattern
    for i in range(n):
        pos = n // 2 + (-1) ** i * ((i + 1) // 2)
        rearranged[pos] = arr_sorted[i]

    # Place nan values at the end
    if np.any(nan_mask):
        rearranged = np.pad(rearranged, (0, nan_mask.sum()), 'constant',
                            constant_values=np.nan)
    return rearranged


def rsquared(y_true, y_pred):
    """
    Calculate the R-squared (coefficient of determination) metric.
    R-squared measures the proportion of variance in the dependent variable
    that is predictable from the independent variable(s).

    Args:
        y_true (array-like): Actual/true values.
        y_pred (array-like): Predicted values.

    Returns:
        float: R-squared value ranging from 0 to 1 (or negative for poor fits).
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def series2func(series: pd.Series, **kwargs) -> interp1d:
    """
    Converts a pandas Series into an interpolating function.

    Args:
        series (pd.Series): Series where index represents the independent
            variable and values the dependent variable.
        **kwargs: Additional arguments for scipy.interpolate.interp1d.

    Returns:
        interp1d: A function that interpolates or extrapolates the
            given series.
    """
    return interp1d(series.index, series.values, fill_value='extrapolate',
                    **kwargs)


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
    except Exception:
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
    except Exception:
        return obj
