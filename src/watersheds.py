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
    def __init__(self, fid, basin, rivers, dem, lulc=[]):
        """
        Drainage Basin class constructor

        Args:
            fid (str): Basin identifier
            basin (GeoDataFrame): Watershed polygon
            rivers (GeoDataFrame): River network segments
            dem (xarray.DataArray): Digital elevation model
            lulc (list): list of additional categorical land cover rasters.

        Raises:
            RuntimeError: If any of the given spatial data isnt in a projected
                cartographic projection.
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
        if len(lulc) != 0:
            for raster in lulc:
                if not raster.rio.crs.is_projected:
                    error = prj_error.format(f'LULC {raster.name} raster')
                    raise RuntimeError(error)
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

        # Properties
        self.params = pd.DataFrame([], index=[self.fid])
        self.hypsometric_curve = pd.Series([])
        self.lulc = lulc

    def __repr__(self) -> str:
        """
        What to show when invoking a RiverBasin object
        Returns:
            str: Some metadata
        """
        text = f'RiverBasin: {self.fid}\n'
        text = text+f'Parameters:\n\n{self.params.head(5)}'
        return text

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

    def process_lulc(self):
        """
        Computes distributions of lulc rasters (% of the basin with the
        X land cover class)

        Returns:
            counts: Percentage of the basin with the X property in each
                    categorical raster.
        """
        try:
            if len(self.lulc) != 0:
                counts = []
                for raster in self.lulc:
                    count = raster.to_series().value_counts()
                    count = count/count.sum()
                    count.name = self.fid
                    count.index = [f'f{raster.name}_{i}' for i in count.index]
                    counts.append(count)
                counts = pd.DataFrame(pd.concat(counts)).T
            else:
                counts = pd.DataFrame([], index=[self.fid])

        except Exception as e:
            counts = pd.DataFrame([], index=[self.fid])
            warnings.warn('LULC rasters Error:', e, self.fid)
        return counts

    def get_lulc_raster(self, name):
        """
        Return a LULC raster from its name

        Args:
            name (str): raster name

        Returns:
            xarray.DataArray:
        """
        for raster in self.lulc:
            if raster.name == name:
                return raster

    def get_lulc_perc(self, name):
        """
        This function takes all computations stored in the "lulc" key 
        of the params dataframe and returns a new pandas object with 
        the raster value in the index and the percentage of the basin with
        that value in the pandas values

        Args:
            name (str): name of the lulc raster to process

        Returns:
            pandas.Series: Series with raster distribution along the basin
        """
        field = self.params.loc['lulc'][self.fid]
        mask = field.index.map(lambda x: name in x)
        field = field.loc[mask]
        values = [v.split(name)[-1].replace('_', '') for v in field.index]
        field.index = values
        field.index = field.index.astype(self.get_lulc_raster(name).dtype)
        return field

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

        # LULC raster distributions
        lulc_counts = self.process_lulc()

        self.params = pd.concat([self.params.T, lulc_counts.T],
                                keys=['geoparams', 'lulc'])

        return self
