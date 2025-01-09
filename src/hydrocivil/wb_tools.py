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

from typing import Tuple
import geopandas as gpd
from shapely.geometry import LineString, Polygon
import warnings
import whitebox_workflows as wbw


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

    da = xr.DataArray(data=wbRaster2numpy(obj),
                      dims=['y', 'x'],
                      coords={'x': ('x', x, {'units': obj.configs.xy_units}),
                              'y': ('y', y, {'units': obj.configs.xy_units})},
                      attrs={'title': obj.configs.title,
                             '_FillValue': obj.configs.nodata,
                             'wkt_code': obj.configs.coordinate_ref_system_wkt,
                             'epsg_code': obj.configs.epsg_code})

    return da


def wbAttributes2DataFrame(obj: wbw.Vector) -> pd.DataFrame:
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
    gdf = gpd.points_from_xy(xs, ys)
    gdf = gpd.GeoDataFrame(geometry=gdf, crs=crs)
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
    from whitebox_workflows import VectorGeometryType
    obj_type = obj.header.shape_type.base_shape_type()
    if obj_type == VectorGeometryType.Point:
        return wbPoint2geopandas(obj, crs=crs)

    elif obj_type == VectorGeometryType.PolyLine:
        return wbLine2geopandas(obj, crs=crs)

    else:  # Polygon
        return wbPolygon2geopandas(obj, crs=crs)


def xarray2wbRasterConfigs(da: xr.DataArray) -> wbw.RasterConfigs:
    """
    Generate basic RasterConfigs from an xarray DataArray.

    Args:
        da (xr.DataArray): Input xarray DataArray containing raster data.

    Returns:
        wbw.RasterConfigs: Configuration object for creating a new raster.
    """
    configs = wbw.RasterConfigs()
    dtype_dict = {'float32': wbw.RasterDataType.F32,
                  'float64': wbw.RasterDataType.F64,
                  'int8': wbw.RasterDataType.I8,
                  'int16': wbw.RasterDataType.I16,
                  'int32': wbw.RasterDataType.I32,
                  'int64': wbw.RasterDataType.I64,
                  '<U8': wbw.RasterDataType.U8,
                  '<U16': wbw.RasterDataType.U16,
                  '<U32': wbw.RasterDataType.U32,
                  '<U64': wbw.RasterDataType.U64}
    # Raster shape
    nrows, ncols = da.shape[0], da.shape[1]
    configs.rows = nrows
    configs.columns = ncols
    bounds = da.rio.bounds()
    configs.west = bounds[0]
    configs.east = bounds[2]
    configs.south = bounds[1]
    configs.north = bounds[3]

    # Raster resolution
    dx, dy = da.rio.resolution()
    configs.resolution_x = abs(dx)
    configs.resolution_y = abs(dy)

    # Projection
    try:
        configs.epsg_code = da.rio.crs.to_epsg()
        configs.coordinate_ref_system_wkt = da.rio.crs.to_wkt()
    except Exception as e:
        warnings.warn(str(e))

    # No data and dtype
    configs.nodata = da.rio.nodata
    configs.data_type = dtype_dict[str(da.dtype)]

    return configs


def xarray2wbRaster(da: xr.DataArray) -> wbw.Raster:
    """
    Convert an xarray DataArray to a WhiteboxTools Raster.

    Args:
        da (xr.DataArray): Input xarray DataArray containing raster data.

    Returns:
        wbw.Raster: A new raster created from the DataArray.
    """
    array = da.values
    configs = xarray2wbRasterConfigs(da)
    new_raster = wbw.WbEnvironment().new_raster(configs)
    for row in range(configs.rows):
        for col in range(configs.columns):
            new_raster[row, col] = array[row, col]
    return new_raster


def wbDEMpreprocess(dem: xr.DataArray,
                    depressions_method: str = 'fill',
                    return_streams: bool = False,
                    flow_accumulation_threshold: float = 1e6
                    ) -> Tuple[xr.Dataset, gpd.GeoDataFrame]:
    """
    Preprocess a DEM (Digital Elevation Model) using WhiteboxTools to create
    a depressionless DEM, compute flow direction, flow accumulation, and flow
    length. Optionally, extract stream networks.

    Args:
        dem (xr.DataArray): Input DEM as an xarray DataArray.
        depressions_method (str, optional): Method to remove depressions.
            Options are 'fill' or 'breach'. Defaults to 'fill'.
        return_streams (bool, optional): Whether to extract and return stream
            networks. Defaults to False.
        flow_accumulation_threshold (float, optional): Threshold for flow
            accumulation to define streams. Defaults to 1e6.
    Returns
        Tuple[xr.Dataset, gpd.GeoDataFrame]: A tuple containing:
            - xr.Dataset: Dataset with flow direction, flow accumulation,
                and flow length.
            - gpd.GeoDataFrame: GeoDataFrame with stream networks if
                return_streams is True, otherwise an empty GeoDataFrame.
    """
    # Create whitebox enviroment and transform dem to a whitebox object
    wbe = wbw.WbEnvironment()
    w_dem = xarray2wbRaster(dem)

    # Create the depressionless DEM
    if depressions_method == 'breach':
        w_dem_filled = wbe.breach_depressions_least_cost(w_dem)
    elif depressions_method == 'fill':
        w_dem_filled = wbe.fill_depressions(w_dem)

    # Compute flow direction, accumulation and length
    w_d8flowdir = wbe.d8_pointer(w_dem_filled)
    w_d8flowacc = wbe.d8_flow_accum(w_d8flowdir, input_is_pointer=True,
                                    out_type='catchment area')
    w_d8flowlen = wbe.downslope_flowpath_length(w_d8flowdir)

    # Transform whitebox objects to xarray data arrays
    d8_flowdir = wbRaster2xarray(w_d8flowdir).to_dataset(name='flowdir')
    d8_flowacc = wbRaster2xarray(w_d8flowacc).to_dataset(name='flowacc')
    d8_flowlen = wbRaster2xarray(w_d8flowlen).to_dataset(name='flowlen')
    rasters = xr.merge([d8_flowdir, d8_flowacc, d8_flowlen])
    rasters = rasters.reindex({'x': dem.x, 'y': dem.y}, method='nearest')
    rasters = rasters.where(~np.isnan(dem))

    # Compute vector streams if asked and return final results
    if return_streams:

        w_streams_r = wbe.extract_streams(w_d8flowacc,
                                          flow_accumulation_threshold)
        w_streams_v0 = wbe.raster_streams_to_vector(w_streams_r, w_d8flowdir)
        w_streams_v0 = wbe.vector_stream_network_analysis(w_streams_v0, w_dem)
        streams_v0 = wbVector2geopandas(w_streams_v0[0])
        return (rasters, streams_v0)
    else:
        return (rasters, gpd.GeoDataFrame())
