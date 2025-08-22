'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 2024-08-05 11:11:38
 Modified by: Lucas Glasner,
 Modified time: 2024-08-05 11:11:43
 Description: Main watershed classes
 Dependencies:
'''
import os
import tempfile

import numpy as np
import pandas as pd
import xarray as xr

import geopandas as gpd
from shapely.geometry import LineString, Polygon
import whitebox_workflows as wbw
wbe = wbw.WbEnvironment(user_id='hydrocivil')

# ---------------------------- wb2py functionality --------------------------- #


def wbRaster2numpy(obj: wbw.Raster) -> np.ndarray:
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

    # Initialize with nodata
    arr = np.full([rows, columns], np.nan)
    for row in range(0, obj.configs.rows):
        arr[row, :] = obj.get_row_data(row)
    return arr


def wbRaster2xarray(obj: wbw.Raster, exchange_rowcol: bool = False,
                    flip_y: bool = False, flip_x: bool = False
                    ) -> xr.DataArray:
    """
    This function grabs a whitebox_workflows Raster object and returns
    the image data as an xarray DataArray.

    Args:
        obj (whitebox_workflows.Raster): A whitebox Raster object
        exchange_rowcol (bool, optional): Whether to flip rows and columns.
            Defaults to False.
        flip_y (bool, optional): Whether to flip the y-axis. Defaults to False.
        flip_x (bool, optional): Whether to flip the x-axis. Defaults to False.

    Returns:
        xr.DataArray: The raster data as an xarray DataArray.
    """
    xstart, xend = obj.configs.west, obj.configs.east
    ystart, yend = obj.configs.south, obj.configs.north
    if exchange_rowcol:
        x = np.linspace(xstart, xend, obj.configs.rows)
        y = np.linspace(ystart, yend, obj.configs.columns)[::-1]
    else:
        x = np.linspace(xstart, xend, obj.configs.columns)
        y = np.linspace(ystart, yend, obj.configs.rows)[::-1]

    if flip_y:
        y = y[::-1]
    if flip_x:
        x = x[::-1]

    x = x+obj.configs.resolution_x/2
    y = y-obj.configs.resolution_y/2

    da = xr.DataArray(data=wbRaster2numpy(obj),
                      dims=['y', 'x'],
                      coords={'x': ('x', x, {'units': obj.configs.xy_units}),
                              'y': ('y', y, {'units': obj.configs.xy_units})},
                      attrs={'title': obj.configs.title,
                             '_FillValue': obj.configs.nodata,
                             'wkt_code': obj.configs.coordinate_ref_system_wkt,
                             'epsg_code': obj.configs.epsg_code})
    da = da.where(da != obj.configs.nodata)

    return da


def wbAttributes2DataFrame(obj: wbw.Vector) -> pd.DataFrame:
    """
    This function grabs a whitebox_workflows vector object and recuperates
    the attribute table as a pandas dataframe.

    Args:
        obj (whitebox_workflows.Vector): A whitebox Vector object

    Returns:
        df (pd.DataFrame): Vector Attribute Table 
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


def wbPoint2geopandas(obj: wbw.Vector, crs: str = None) -> gpd.GeoDataFrame:
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
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xs, ys), crs=crs)
    gdf_attrs = wbAttributes2DataFrame(obj)
    gdf = pd.concat([gdf_attrs, gdf], axis=1).set_geometry('geometry')
    return gdf


def wbLine2geopandas(obj: wbw.Vector, crs: str = None) -> gpd.GeoDataFrame:
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


def wbPolygon2geopandas(obj: wbw.Vector, crs: str = None) -> gpd.GeoDataFrame:
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


def wbVector2geopandas(obj: wbw.Vector, crs: str = None) -> gpd.GeoDataFrame:
    """
    This function transform a whitebox_workflows vector layer to a geopandas
    GeoDataFrame.

    Args:
        obj (whitebox_workflows.Vector): A whitebox Vector object
        crs (str, optional): Cartographic Projection. Defaults to None.

    Returns:
        gdf (geopandas.GeoDataFrame): Vector layer as a GeoDataFrame object
    """
    if obj is None:
        return gpd.GeoDataFrame()
    from whitebox_workflows import VectorGeometryType
    obj_type = obj.header.shape_type.base_shape_type()
    if obj_type == VectorGeometryType.Point:
        return wbPoint2geopandas(obj, crs=crs)

    elif obj_type == VectorGeometryType.PolyLine:
        return wbLine2geopandas(obj, crs=crs)

    else:  # Polygon
        return wbPolygon2geopandas(obj, crs=crs)


def xarray2wbRaster(da: xr.DataArray) -> wbw.Raster:
    """
    Convert an xarray DataArray to a WhiteboxTools Raster.

    Args:
        da (xr.DataArray): Input xarray DataArray containing raster data.

    Returns:
        wbw.Raster: A new raster created from the DataArray.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        fpath = os.path.join(tmpdirname, f'{os.path.basename(tmpdirname)}.tif')
        da.attrs.pop('_FillValue', None)
        da.rio.to_raster(fpath)
        wda = wbe.read_raster(fpath)
    return wda


def geopandas2wbVector(gdf: gpd.GeoDataFrame) -> wbw.Vector:
    """
    Convert a GeoDataFrame to a WhiteboxTools Vector.

    Args:
        gdf (gpd.GeoDataFrame): Input GeoDataFrame containing vector data.

    Returns:
        wbw.Vector: A new vector created from the GeoDataFrame.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        fpath = os.path.join(tmpdirname, f'{os.path.basename(tmpdirname)}.shp')
        gdf.to_file(fpath)
        wv = wbe.read_vector(fpath)
    return wv
