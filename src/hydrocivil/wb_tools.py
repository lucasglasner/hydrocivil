'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 2024-08-05 11:11:38
 Modified by: Lucas Glasner,
 Modified time: 2024-08-05 11:11:43
 Description: Main watershed classes
 Dependencies:
'''

import numpy as np
import pandas as pd
import xarray as xr

from typing import Any
import geopandas as gpd
from shapely.geometry import LineString, Polygon


def wbRaster2numpy(obj: Any) -> np.ndarray:
    """
    This function grabs a whitebox_workflows Raster object and return
    the image data as a numpy array

    Args:
        obj (whitebox_workflows.Raster): A whitebox Raster object

    Returns:
        (numpy.array): data
    """
    rows = int(np.ceil(obj.configs.rows))
    columns = int(np.ceil(obj.configs.columns))
    nodata = obj.configs.nodata

    # Initialize with nodata
    arr = np.full([rows, columns], np.nan)
    r = 0
    for row in range(0, obj.configs.rows):
        values = obj.get_row_data(row)
        c = 0
        for col in range(0, obj.configs.columns):
            value = values[col]
            if value != nodata:
                arr[r, c] = value
            c += 1
        r += 1
    return arr


def wbRaster2xarray(obj: Any, flip_y: bool = False, flip_x: bool = False
                    ) -> xr.DataArray:
    """
    This function grabs a whitebox_workflows Raster object and return
    the image data as an xarray DataArray

    Args:
        obj (whitebox_workflows.Raster): A whitebox Raster object

    Returns:
        (xarray.DataArray): data
    """
    xstart, xend = obj.configs.west, obj.configs.east
    ystart, yend = obj.configs.south, obj.configs.north
    dx, dy = obj.configs.resolution_x, obj.configs.resolution_y
    x = np.arange(xstart, xend+dx, dx)
    y = np.arange(ystart, yend+dy, dy)[::-1]

    if flip_y:
        y = y[::-1]
    if flip_x:
        x = x[::-1]

    da = xr.DataArray(data=wbRaster2numpy(obj),
                      dims=['y', 'x'],
                      coords={'x': ('x', x, {'units': obj.configs.xy_units}),
                              'y': ('y', y, {'units': obj.configs.xy_units})},
                      attrs={'title': obj.configs.title,
                             '_FillValue': obj.configs.nodata,
                             'wkt_code': obj.configs.coordinate_ref_system_wkt,
                             'epsg_code': obj.configs.epsg_code})

    return da


def wbAttributes2DataFrame(obj: Any) -> pd.DataFrame:
    """
    This function grabs a whitebox_workflows vector object and recuperates
    the attribute table as a pandas dataframe.

    Args:
        obj (whitebox_workflows.Vector): A whitebox Vector object

    Returns:
        df (pandas.DataFrame): Vector Attribute Table 
    """
    attrs = obj.attributes.fields
    names = [field.name for field in attrs]

    df = []
    for c in names:
        values = []
        for i in range(obj.num_records):
            val = obj.get_attribute_value(i, c)
            values.append(val)
        values = pd.Series(values, index=range(obj.num_records), name=c)
        df.append(values)

    df = pd.concat(df, axis=1)
    return df


def wbPoint2geopandas(obj: Any, crs: str = None) -> gpd.GeoDataFrame:
    """
    This function transform a whitebox_workflows Point layer to a geopandas
    GeoDataFrame.

    Args:
        obj (whitebox_workflows.Vector): A whitebox vector object with points
        crs (str, optional): Cartographic Projection. Defaults to None.

    Returns:
        gdf (geopandas.GeoDataFrame): Point layer as a GeoDataFrame
    """
    xs = []
    ys = []
    for rec in obj:
        x, y = rec.get_xy_data()
        xs.append(x)
        ys.append(y)
    xs, ys = np.array(xs).squeeze(), np.array(ys).squeeze()
    gdf = gpd.points_from_xy(xs, ys)
    gdf = gpd.GeoDataFrame(geometry=gdf, crs=crs)
    gdf_attrs = wbAttributes2DataFrame(obj)
    gdf = pd.concat([gdf_attrs, gdf], axis=1).set_geometry('geometry')
    return gdf


def wbLine2geopandas(obj: Any, crs: str = None) -> gpd.GeoDataFrame:
    """
    This function transform a whitebox_workflows Line layer to a geopandas
    GeoDataFrame.

    Args:
        obj (whitebox_workflows.Vector): A whitebox vector object with lines
        crs (str, optional): Cartographic Projection. Defaults to None.

    Returns:
        gdf (geopandas.GeoDataFrame): Lines as a GeoDataFrame object
    """
    xs = []
    ys = []
    for rec in obj:
        parts = rec.parts
        num_parts = rec.num_parts
        part_num = 1  # actually the next part
        x, y = rec.get_xy_data()
        for i in range(len(x)):
            if part_num < num_parts and i == parts[part_num]:
                xs.append(np.nan)  # discontinuity
                ys.append(np.nan)  # discontinuity
                part_num += 1

            xs.append(x[i])
            ys.append(y[i])
        xs.append(np.nan)  # discontinuity
        ys.append(np.nan)  # discontinuity
    xs, ys = np.array(xs).squeeze(), np.array(ys).squeeze()

    breaks = np.where(np.isnan(xs))[0]
    slices = [slice(None, breaks[0])]
    for i in range(len(breaks)-1):
        slices.append(slice(breaks[i]+1, breaks[i+1]))

    lines = []
    for s in slices:
        line = LineString([(x, y) for x, y in zip(xs[s], ys[s])])
        lines.append(line)

    gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)
    gdf_attrs = wbAttributes2DataFrame(obj)
    gdf = pd.concat([gdf_attrs, gdf], axis=1).set_geometry('geometry')

    return gdf


def wbPolygon2geopandas(obj: Any, crs: str = None) -> gpd.GeoDataFrame:
    """
    This function transform a whitebox_workflows Polygon layer to a geopandas
    GeoDataFrame.

    Args:
        obj (whitebox_workflows.Vector): A whitebox vector object with polygons
        crs (str, optional): Cartographic Projection. Defaults to None.

    Returns:
        gdf (geopandas.GeoDataFrame): Polygons as a GeoDataFrame object
    """
    xs = []
    ys = []
    for rec in obj:
        parts = rec.parts
        num_parts = rec.num_parts
        part_num = 1  # actually the next part
        x, y = rec.get_xy_data()
        for i in range(len(x)):
            if part_num < num_parts and i == parts[part_num]:
                xs.append(np.nan)  # discontinuity
                ys.append(np.nan)  # discontinuity
                part_num += 1

            xs.append(x[i])
            ys.append(y[i])

        xs.append(np.nan)  # discontinuity
        ys.append(np.nan)  # discontinuity

    xs, ys = np.array(xs).squeeze(), np.array(ys).squeeze()

    breaks = np.where(np.isnan(xs))[0]
    slices = [slice(None, breaks[0])]
    for i in range(len(breaks)-1):
        slices.append(slice(breaks[i]+1, breaks[i+1]))

    poly = []
    for s in slices:
        line = Polygon([(x, y) for x, y in zip(xs[s], ys[s])])
        poly.append(line)

    gdf = gpd.GeoDataFrame(geometry=poly, crs=crs)
    gdf_attrs = wbAttributes2DataFrame(obj)
    gdf = pd.concat([gdf_attrs, gdf], axis=1).set_geometry('geometry')

    return gdf


def wbVector2geopandas(obj: Any, crs: str = None) -> gpd.GeoDataFrame:
    """
    This function transform a whitebox_workflows vector layer to a geopandas
    GeoDataFrame.

    Args:
        obj (whitebox_workflows.Vector): A whitebox Vector object
        crs (str, optional): Cartographic Projection. Defaults to None.

    Returns:
        gdf (geopandas.GeoDataFrame): Vector layer as a GeoDataFrame object
    """
    from whitebox_workflows import VectorGeometryType
    obj_type = obj.header.shape_type.base_shape_type()
    if obj_type == VectorGeometryType.Point:
        return wbPoint2geopandas(obj, crs=crs)

    elif obj_type == VectorGeometryType.PolyLine:
        return wbLine2geopandas(obj, crs=crs)

    else:  # Polygon
        return wbPolygon2geopandas(obj, crs=crs)
