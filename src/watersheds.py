'''
 # @Author: Lucas Glasner (lgvivanco96@gmail.com)
 # @Create Time: 2024-08-05 11:11:38
 # @Modified by: Lucas Glasner,
 # @Modified time: 2024-08-05 11:11:43
 # @Description: Main Basin/watershed class
 # @Dependencies:

'''

import os
import warnings

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

from osgeo import gdal
from scipy.interpolate import interp1d

from src.geomorphology import *
from src.misc import get_psep, raster_distribution

# ---------------------------------------------------------------------------- #


class RiverBasin(object):
    def tests(self, basin, rivers, dem, cn):
        """
        Args:
            basin (GeoDataFrame): Basin polygon
            rivers (GeoDataFrame): River network lines
            dem (xarray.DataArray): Digital elevation model raster
            cn (xarray.DataArray): Curve number raster
        Raises:
            RuntimeError: If any dataset isnt in a projected (UTM) crs.
        """
        prj_error = '{} must be in a projected (UTM) crs !'
        if not basin.crs.is_projected:
            error = prj_error.format('Watershed geometry')
            raise RuntimeError(error)
        if not rivers.crs.is_projected:
            error = prj_error.format('Rivers geometry')
            raise RuntimeError(error)
        if not dem.rio.crs.is_projected:
            error = prj_error.format('DEM raster')
            raise RuntimeError(error)
        if type(cn) != type(None):
            if not cn.rio.crs.is_projected:
                error = prj_error.format('Curve Number raster')
                raise RuntimeError(error)

    def __init__(self, fid, basin, rivers, dem, cn=None):
        """
        Drainage Basin class constructor

        Args:
            fid (str): Basin identifier
            basin (GeoDataFrame): Watershed polygon
            rivers (GeoDataFrame): River network segments
            dem (xarray.DataArray): Digital elevation model
            cn (xarray.DataArray): Curve Number raster. Defaults to None.

        Raises:
            RuntimeError: If any of the given spatial data isnt in a projected
                cartographic projection.
        """
        self.tests(basin, rivers, dem, cn)
        # ID
        self.fid = fid

        # Vectors
        self.basin = basin
        self.rivers = rivers
        self.rivers_main = pd.Series([])

        # Terrain
        self.dem = dem.rio.write_nodata(-9999).squeeze()
        self.dem = self.dem.to_dataset(name='elevation')
        self.dem.encoding = dem.encoding

        # Curve Number
        if type(cn) != type(None):
            cn = cn.rio.write_nodata(-9999).squeeze()
            self.cn = cn
        else:
            self.cn = None

        # Properties
        self.params = pd.DataFrame([], index=[self.fid])
        self.hypsometric_curve = pd.Series([])

    def __repr__(self) -> str:
        """
        What to show when invoking a RiverBasin object
        Returns:
            str: Some metadata
        """
        text = f'RiverBasin: {self.fid}\n'
        text = text+f'Parameters:\n\n{self.params.head(5)}'
        return text

    def set_parameter(self, index, value):
        """
        Simple function to add a new parameter to the basin parameters table

        Args:
            index (str): parameter name/id or what to put in the table index
            value (object): value of the new parameter
        """
        self.params.loc[index, :] = value
        return self

    def process_gdaldem(self, varname, open_rasterio_kwargs={}, **kwargs):
        """
        Accessor to gdaldem command line utility.

        Args:
            open_rasterio_kwargs (dict, optional):
                Arguments for rioxarray open rasterio function. Defaults to {}.

        Returns:
            xarray.DataArray: DEM derived property
        """
        psep = get_psep()
        iname = self.dem.elevation.encoding['source']
        oname = self.dem.elevation.encoding['source'].split(psep)
        oname = f'{psep}'.join(oname[:-1])+f'{psep}{varname}_'+oname[-1]
        gdal.DEMProcessing(oname, iname, varname, **kwargs)
        field = rxr.open_rasterio(oname, **open_rasterio_kwargs)
        field = field.squeeze().to_dataset(name=varname)
        return field

    def process_hypsometric_curve(self, bins='auto', **kwargs):
        """
        Based on terrain, compute hypsometric curve of the basin

        Returns:
            pandas.Series: Hypsometric curve expressed as fraction of area
                           below a certain elevation.
        """
        curve = raster_distribution(self.dem.elevation, bins=bins, **kwargs)
        self.hypsometric_curve = curve.cumsum()
        return curve

    def area_below_height(self, height, **kwargs):
        """
        With the hypsometric curve compute the fraction of area below
        a certain height in the basin.

        Args:
            height (float): elevation value

        Returns:
            (float): fraction of area below given elevation
        """
        if len(self.hypsometric_curve) == 0:
            warnings.warn('Computing hypsometric curve ...')
            self.process_hypsometric_curve(**kwargs)
        curve = self.hypsometric_curve
        if height < curve.index.min():
            return 0
        if height > curve.index.max():
            return 1
        interp_func = interp1d(curve.index.values, curve.values)
        return interp_func(height).item()

    def process_geography(self):
        """
        Compute geographical parameters of the basin

        Returns:
            self: updated class
        """
        try:
            geo_params = basin_geographical_params(self.fid, self.basin)
        except Exception as e:
            geo_params = pd.DataFrame([], index=[self.fid])
            warnings.warn('Geographical Parameters Error:', e, self.fid)
        self.params = pd.concat([self.params, geo_params], axis=1)
        return self

    def process_dem(self, **kwargs):
        """
        Compute hypsometric curve, slope and aspect. Then compute DEM
        derived propeties for the basin and save in the params dataframe.

        Returns:
            self: updated class
        """
        try:
            curve = self.process_hypsometric_curve()
            slope = self.process_gdaldem('slope',
                                         computeEdges=True,
                                         slopeFormat='percent',
                                         open_rasterio_kwargs={**kwargs})
            aspect = self.process_gdaldem('aspect',
                                          computeEdges=True,
                                          open_rasterio_kwargs={**kwargs})
            self.dem = xr.merge([self.dem, slope.copy()/100, aspect.copy()])
            self.dem.attrs = {'standard_name': 'terrain model',
                              'hypsometry_x': [f'{i:.2f}' for i in curve.index],
                              'hypsometry_y': [f'{j:3f}' for j in curve.values]}
            # DEM derived params
            terrain_params = basin_terrain_params(self.fid, self.dem)
        except Exception as e:
            terrain_params = pd.DataFrame([], index=[self.fid])
            warnings.warn('PostProcess DEM Error:', e, self.fid)
        self.params = pd.concat([self.params, terrain_params], axis=1)
        return self

    def process_river_network(self, **kwargs):
        """
        Compute river network properties

        Returns:
            self: updated class
        """
        try:
            mainriver = main_river(self.rivers, **kwargs)
            self.rivers_main = self.rivers.loc[mainriver.index]
            mriverlen = self.rivers_main.length.sum()/1e3
            if mriverlen.item() != 0:
                mriverlen = mriverlen.item()
            else:
                mriverlen = np.nan
            self.params['mriverlen_km'] = mriverlen
            rhod = self.rivers.length.sum()/self.params['area_km2']/1e3
            Kf = self.params['area_km2']/(self.params['mriverlen_km']**2)
            self.params['rhod_1'] = rhod
            self.params['Kf_1'] = Kf
        except Exception as e:
            warnings.warn('Flow derived properties Error:', e, self.fid)
        return self

    def process_raster_counts(self, raster, output_type=1):
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
            counts.name = self.fid
            if output_type == 1:
                counts = counts.reset_index()
            elif output_type == 2:
                counts.index = [f'f{raster.name}_{i}' for i in counts.index]
                counts = pd.DataFrame(counts)
            else:
                raise RuntimeError(f'{output_type} must only be 1 or 2.')
        except Exception as e:
            counts = pd.DataFrame([], columns=[self.fid],
                                  index=[0])
            warnings.warn('Raster counting Error:', e, self.fid)
        return counts

    def compute_params(self,
                       main_river_kwargs={},
                       gdal_kwargs={}):
        """
        Compute basin geomorphological properties:
            1) Geographical properties: centroid coordinates, area, etc
                Details in src.geomorphology.basin_geographical_params routine
            2) Terrain properties: DEM derived properties like minimum, maximum
                or mean height, etc.
                Details in src.geomorphology.basin_terrain_params
            3) Flow derived properties: Main river length using graph theory, 
                drainage density and shape factor. 
                Details in src.geomorphology.main_river
        Args:
            main_river_kwargs (dict, optional): 
                Additional arguments for the main river finding routine.
                Defaults to {}. Details in src.geomorphology.main_river routine
            gdal_kwargs (dict, optional): 
                Additional arguments for the slope computing function.
                Defaults to {}.

        Returns:
            pandas.DataFrame: Table with basin geomorphological properties
        """
        if self.params.shape != (1, 0):
            self.params = pd.DataFrame([], index=[self.fid])

        # Geographical parameters
        self.process_geography()

        # Compute slope and aspect. Update dem property
        self.process_dem(masked=True, **gdal_kwargs)

        # Flow derived params
        self.process_river_network(**main_river_kwargs)

        # Curve number process
        if type(self.cn) != type(None):
            cn_counts = self.process_raster_counts(self.cn, output_type=2)
            cn_counts = cn_counts[self.fid]
            cn_counts.loc['curvenumber_1'] = self.cn.mean().item()
        else:
            cn_counts = pd.DataFrame([])

        self.params = pd.concat([self.params.T, cn_counts],
                                keys=['geoparams', 'lulc'])

        return self
