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
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from typing import Tuple
import geopandas as gpd
from shapely.geometry import LineString, Polygon
import whitebox_workflows as wbw
wbe = wbw.WbEnvironment(user_id='hydrocivil')


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
    from whitebox_workflows import VectorGeometryType
    obj_type = obj.header.shape_type.base_shape_type()
    if obj_type == VectorGeometryType.Point:
        return wbPoint2geopandas(obj, crs=crs)

    elif obj_type == VectorGeometryType.PolyLine:
        return wbLine2geopandas(obj, crs=crs)

    else:  # Polygon
        return wbPolygon2geopandas(obj, crs=crs)


# def xarray2wbRasterConfigs(da: xr.DataArray) -> wbw.RasterConfigs:
#     """
#     Generate basic RasterConfigs from an xarray DataArray.

#     Args:
#         da (xr.DataArray): Input xarray DataArray containing raster data.

#     Returns:
#         wbw.RasterConfigs: Configuration object for creating a new raster.
#     """
#     configs = wbw.RasterConfigs()
#     dtype_dict = {'float32': wbw.RasterDataType.F32,
#                   'float64': wbw.RasterDataType.F64,
#                   'int8': wbw.RasterDataType.I8,
#                   'int16': wbw.RasterDataType.I16,
#                   'int32': wbw.RasterDataType.I32,
#                   'int64': wbw.RasterDataType.I64,
#                   '<U8': wbw.RasterDataType.U8,
#                   '<U16': wbw.RasterDataType.U16,
#                   '<U32': wbw.RasterDataType.U32,
#                   '<U64': wbw.RasterDataType.U64}
#     # Raster shape
#     nrows, ncols = da.shape[0], da.shape[1]
#     configs.rows = nrows
#     configs.columns = ncols
#     bounds = da.rio.bounds()
#     configs.west = bounds[0]
#     configs.east = bounds[2]
#     configs.south = bounds[1]
#     configs.north = bounds[3]

#     # Raster resolution
#     dx, dy = da.rio.resolution()
#     configs.resolution_x = abs(dx)
#     configs.resolution_y = abs(dy)

#     # Projection
#     try:
#         configs.epsg_code = da.rio.crs.to_epsg()
#         configs.coordinate_ref_system_wkt = da.rio.crs.to_wkt()
#     except Exception as e:
#         warnings.warn(str(e))

#     # No data and dtype
#     configs.nodata = np.nan
#     configs.data_type = dtype_dict[str(da.dtype)]

#     return configs


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
        da.rio.to_raster(fpath)
        wda = wbe.read_raster(fpath).deep_copy()
    return wda


def wbDEMflow(dem_no_deps: wbw.Raster | xr.DataArray,
              method: str = 'd8',
              input_is_xarray: bool = False
              ) -> Tuple[wbw.Raster | xr.DataArray, wbw.Raster | xr.DataArray]:
    """
    Given a depresionless DEM this function computes flow direction and 
    flow accumulation with different methods.
    Args:
        dem_no_deps (wbw.Raster, xr.DataArray): Input depresionless DEM. 
        flow_method (str, optional): Flow direction algorithm used for
            computing flow direction and flow accumulation rasters. 
            Defaults to 'd8'. Options include:
                'd8', 'rho8', 'dinf', 'fd8', 'Mdinf', 'Quinn1995', 'Qin2007'.
        input_is_xarray (bool, optional): Whether to transform the input
            xarray object to a whitebox_workflows Raster. Defaults to False.

    Raises:
        ValueError: If given an unknown flow computation method.

    Returns:
        (tuple): flow direction raster, flow accumulation raster
    """
    if input_is_xarray:
        dem_no_deps = xarray2wbRaster(dem_no_deps)

    if method == 'd8':
        fdir = wbe.d8_pointer(dem_no_deps)
        facc = wbe.d8_flow_accum(fdir, input_is_pointer=True,
                                 out_type='catchment area')
    elif method == 'rho8':
        fdir = wbe.rho8_pointer(dem_no_deps)
        facc = wbe.rho8_flow_accum(fdir, input_is_pointer=True,
                                   out_type='catchment area')
    elif method == 'dinf':
        fdir = wbe.dinf_pointer(dem_no_deps)
        facc = wbe.dinf_flow_accum(fdir, input_is_pointer=True,
                                   out_type='catchment area')
    elif method == 'fd8':
        fdir = wbe.fd8_pointer(dem_no_deps)
        facc = wbe.fd8_flow_accum(dem_no_deps,
                                  out_type='catchment area')
    elif method == 'Mdinf':
        fdir = wbe.dinf_pointer(dem_no_deps)
        facc = wbe.mdinf_flow_accum(dem_no_deps,
                                    out_type='catchment area')
    elif method == 'Quinn1995':
        fdir = wbe.fd8_pointer(dem_no_deps)
        facc = wbe.quinn_flow_accumulation(dem_no_deps,
                                           out_type='catchment area')
    elif method == 'Qin2007':
        fdir = wbe.fd8_pointer(dem_no_deps)
        facc = wbe.qin_flow_accumulation(dem_no_deps,
                                         out_type='catchment area')
    else:
        text = f"'{method}': Unknown flow direction method!"
        raise ValueError(text)

    if input_is_xarray:
        fdir = wbRaster2xarray(fdir).to_dataset(name='fdir')
        facc = wbRaster2xarray(facc).to_dataset(name='facc')

    return fdir, facc


def wbDEMfill(dem: wbw.Raster | xr.DataArray,
              input_is_xarray: bool = False,
              carve_dist: float = 0,
              fill_kws: dict = {},
              breach_kws: dict = {}):
    """

    Args:
        dem (wbw.Raster, xr.DataArray): Input digital elevation model.
        input_is_xarray (bool, optional): Whether to transform the input
            xarray object to a whitebox_workflows Raster. Defaults to False.
        carve_dist (float, optional): Maximum distance to carve when breaching.
            Defaults to 0.
        fill_kws (dict, optional): Additional arguments for the fill
            depressions method.
        breach_kws (dict, optional): Additional arguments for the breach
            depressions method.

    Returns:
        (tuple): smoothed DEM raster, hillshade raster, sinks raster and
                 depresionless DEM raster.
    """
    if input_is_xarray:
        dem = xarray2wbRaster(dem)

    # Compute sinks
    sinks = wbe.sink(dem)

    # Create the depressionless DEM
    if carve_dist != 0:
        dem_no_deps = wbe.breach_depressions_least_cost(dem,
                                                        max_dist=carve_dist,
                                                        **breach_kws)
        dem_no_deps = wbe.fill_depressions(dem_no_deps, **fill_kws)
    else:
        dem_no_deps = wbe.fill_depressions(dem, **fill_kws)

    if input_is_xarray:
        dem = wbRaster2xarray(dem).to_dataset(name='elevation_smooth')
        sinks = wbRaster2xarray(sinks).to_dataset(name='sinks')
        dem_no_deps = wbRaster2xarray(dem_no_deps).to_dataset(
            name='elevation_nodeps')
    return (sinks, dem_no_deps)


def wbDEMstreams(dem: wbw.Raster,
                 fdir: wbw.Raster,
                 facc: wbw.Raster,
                 facc_threshold: float = 1e6):
    """
    Extracts stream networks from a DEM using flow direction and flow
    accumulation rasters.
    Args:
        dem (wbw.Raster): Digital Elevation Model raster.
        fdir (wbw.Raster): Flow direction raster.
        facc (wbw.Raster): Flow accumulation raster.
        facc_threshold (float, optional): Threshold for flow
            accumulation to define streams. Defaults to 1e6 (1 km2).
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing the extracted stream
            network.
    """
    streams_r = wbe.extract_streams(facc, facc_threshold)
    streams_v = wbe.raster_streams_to_vector(streams_r, fdir)
    streams_v = wbe.vector_stream_network_analysis(streams_v, dem)[0]
    return streams_v, streams_r


def wbDEMpreprocess(dem: xr.DataArray,
                    raster2xarray: bool = False,
                    vector2geopandas: bool = False,
                    carve_dist: float = 0,
                    flow_method: str = 'd8',
                    return_streams: bool = False,
                    facc_threshold: float = 1e5,
                    fill_kws: dict = {},
                    breach_kws: dict = {}) -> Tuple[xr.Dataset, gpd.GeoDataFrame]:
    """
    Preprocess a DEM (Digital Elevation Model) using WhiteboxTools to create
    a depressionless DEM, compute flow direction, flow accumulation, and flow
    length. Optionally, extract stream networks.

    Args:
        dem (xr.DataArray): Input DEM as an xarray DataArray.
        raster2xarray (bool, optional): Whether to transform output rasters
            to xarray objects. Defaults to False.
        vector2geopandas (bool, optional): Whether to transform output vectors
            to geopandas objects. Defaults to False.
        carve_dist (float, optional): Maximum distance to carve when breaching.
            Defaults to 0.
        flow_method (str, optional): Flow direction algorithm used for
            computing flow direction and flow accumulation rasters. Defaults to
            'd8'. Options include: 'd8', 'rho8', 'dinf', 'fd8', 'Mdinf',
            'Quinn1995', 'Qin2007'.
        return_streams (bool, optional): Whether to extract and return stream
            networks. Defaults to False.
        facc_threshold (float, optional): Threshold for flow
            accumulation to define streams. Defaults to 1e5.
        fill_kws (dict, optional): Additional arguments for the fill
            depressions method.
        breach_kws (dict, optional): Additional arguments for the breach
            depressions method.
    Returns
        Tuple[xr.Dataset, gpd.GeoDataFrame]: A tuple containing:
            - xr.Dataset: Dataset with flow direction, flow accumulation,
                and flow length.
            - gpd.GeoDataFrame: GeoDataFrame with stream networks if
                return_streams is True, otherwise an empty GeoDataFrame.
    """
    def _getf64(obj):
        """
        Simple function to return whitebox object datatype as float64
        """
        try:
            return obj.get_value_as_f64()
        except Exception:
            return obj
    dem_x = dem.copy()
    dem = xarray2wbRaster(dem)

    # DEM preprocess sinks
    sinks, dem_no_deps = wbDEMfill(dem, carve_dist=carve_dist,
                                   fill_kws=fill_kws, breach_kws=breach_kws)

    # Compute flow direction, accumulation and flow path length
    fdir, facc = wbDEMflow(dem_no_deps, method=flow_method)

    # Join rasters
    names = ['elevation_nodeps', 'sinks', 'fdir', 'facc']
    rasters = [dem_no_deps, sinks, fdir, facc]

    # Compute vector streams if asked and return final results
    if return_streams:
        streams_v, streams_r = wbDEMstreams(dem, fdir, facc,
                                            facc_threshold=facc_threshold)
        names.append('streams')
        rasters.append(streams_r)
    else:
        streams_v = None

    # Transform whitebox Raster objects to xarray data arrays
    if raster2xarray:
        nrasters = []
        for n, da in zip(names, rasters):
            da = wbRaster2xarray(da).to_dataset(name=n)
            da = da.reindex(x=dem_x.x, y=dem_x.y, method='nearest')
            da = da.where(~dem_x.isnull()).rio.write_crs(dem_x.rio.crs)
            nrasters.append(da)
        rasters = nrasters

    if vector2geopandas:
        streams_v = wbVector2geopandas(streams_v).map(lambda x: _getf64(x))

    return (rasters, streams_v)
