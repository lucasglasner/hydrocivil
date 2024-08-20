'''
 # @Author: Lucas Glasner (lgvivanco96@gmail.com)
 # @Create Time: 2024-08-05 11:11:38
 # @Modified by: Lucas Glasner,
 # @Modified time: 2024-08-05 11:11:43
 # @Description: Main Basin/watershed class
 # @Dependencies:

'''

import os
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from osgeo import gdal
from scipy.interpolate import interp1d

from src.geomorphology import *
from src.misc import get_psep


class DrainageBasin:
    def __init__(self, fid, wgeometry, rgeometry, dem, aux_rasters=[]):
        """
        Drainage Basin class constructor

        Args:
            fid (str): Basin identifier
            wgeometry (GeoDataFrame): Watershed/Basin polygon
            rgeometry (GeoDataFrame): RiverNetwork segments
            dem (xarray.DataArray): Digital elevation model
            aux_rasters (list): list of additional xarray.DataArray rasters

        Raises:
            RuntimeError: If any of the given spatial data isnt in a projected
                cartographic projection.
        """
        if not wgeometry.crs.is_projected:
            error = 'Watershed geometry must be in a projected (UTM) crs !'
            raise RuntimeError(error)
        if not rgeometry.crs.is_projected:
            error = 'River network geometry must be in a projected (UTM) crs !'
            raise RuntimeError(error)
        # ID
        self.fid = fid

        # Vectors
        self.wgeometry = wgeometry
        self.rgeometry = rgeometry

        # Terrain
        self.dem = dem.rio.write_nodata(-9999).squeeze()
        self.dem = self.dem.to_dataset(name='elevation')
        self.dem.encoding = dem.encoding

        # Properties
        self.params = pd.DataFrame([], index=[self.fid])
        self.hypsometric_curve = pd.Series([])
        self.aux_rasters = aux_rasters

    def __repr__(self) -> str:
        """
        What to show when invoking a DrainageBasin object
        Returns:
            str: Some metadata
        """
        text = f'DrainageBasin\nFID: {self.fid}\n{self.params}'
        return text

    def raster_distribution(self, raster, **kwargs):
        """

        Args:
            raster (_type_): _description_

        Returns:
            (tuple): _description_
        """
        total_pixels = raster.size
        total_pixels = total_pixels-np.isnan(raster).sum()
        total_pixels = total_pixels.item()

        values = raster.squeeze().values.flatten()
        values = values[~np.isnan(values)]
        dist, values = np.histogram(values, **kwargs)
        dist, values = dist/total_pixels, 0.5*(values[:-1]+values[1:])
        return pd.Series(dist, index=values, name=f'{raster.name}_dist')

    def compute_hypsometry(self, bins='auto', **kwargs):
        """
        Based on terrain, compute hypsometric curve of the basin

        Returns:
            pandas.Series: Hypsometric curve expressed as fraction of area
                           below a certain elevation.
        """
        curve = self.raster_distribution(self.dem.elevation,
                                         bins=bins, **kwargs)
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
        self.compute_hypsometry(**kwargs)
        curve = self.hypsometric_curve
        if height < curve.min():
            return 0
        if height > curve.max():
            return 1
        else:
            interp_func = interp1d(curve.index, curve)
            return interp_func(height)

    def compute_slope(self, open_rasterio_kwargs={},
                      gdaldem_kwargs={}
                      ):
        """
        Computes slope raster from DEM using standard gdal routines

        Args:
            gdaldem_kwargs (dict, optional):
                Arguments for gdal DEMProcessing algorithm. Defaults to {}.
            open_rasterio_kwargs (dict, optional):
                Arguments for rioxarray open rasterio function. Defaults to {}.

        Returns:
            xarray.DataArray: DEM derived slopes in m/m
        """
        args = ' '.join([f'-{i} {j}' for i, j in gdaldem_kwargs.items()])
        psep = get_psep()
        iname = self.dem.elevation.encoding['source']
        oname = self.dem.elevation.encoding['source'].split(psep)
        oname = f'{psep}'.join(oname[:-1])+f'{psep}SLOPE_'+oname[-1]
        gdal.DEMProcessing(oname, iname, 'slope',
                           computeEdges=True,
                           slopeFormat='percent',
                           **gdaldem_kwargs)

        # command = 'gdaldem slope {} {} {}'
        # command = command.format(iname, oname, args)
        # out = os.popen(command)
        try:
            slope = rxr.open_rasterio(
                oname, masked=True, **open_rasterio_kwargs)
            slope = slope.squeeze().to_dataset(name='slope')
            slope = slope.where(slope != -9999)/100
        except Exception as e:
            raise RuntimeError(f'Error: {e}')
        return slope

    def compute_aspect(self, open_rasterio_kwargs={},
                       gdaldem_kwargs={}):
        """
        Computes aspect raster from DEM using standard gdal routines

        Args:
            gdaldem_kwargs (dict, optional):
                Arguments for gdal DEMProcessing algorithm. Defaults to {}.
            open_rasterio_kwargs (dict, optional):
                Arguments for rioxarray open rasterio function. Defaults to {}.

        Returns:
            xarray.DataArray: DEM derived aspect in degrees
        """
        args = ' '.join([f'-{i} {j}' for i, j in gdaldem_kwargs.items()])
        psep = get_psep()
        iname = self.dem.elevation.encoding['source']
        oname = self.dem.elevation.encoding['source'].split(psep)
        oname = f'{psep}'.join(oname[:-1])+f'{psep}ASPECT_'+oname[-1]
        gdal.DEMProcessing(oname, iname, 'aspect', computeEdges=True,
                           **gdaldem_kwargs)
        # command = 'gdaldem aspect {} {} {}'
        # command = command.format(iname, oname, args)
        # out = os.popen(command)

        try:
            aspect = rxr.open_rasterio(
                oname, masked=True, **open_rasterio_kwargs)
            aspect = aspect.squeeze().to_dataset(name='aspect')
            aspect = aspect.where(aspect != -9999)
        except Exception as e:
            raise RuntimeError(f'Error: {e}')
        return aspect

    def compute_params(self,
                       main_river_kwargs={},
                       slope_gdal_kwargs={},
                       aspect_gdal_kwargs={}):
        """
        Compute basin geomorphological properties:
            1) Geographical properties: centroid coordinates, area, etc
                Details in src.geomorphology.basin_geographical_params routine
            2) Terrain properties: DEM derived properties like minimum, maximum
                or mean height, etc.
                Detalils in src.geomorphology.basin_terrain_params
            3) Flow derived properties: Main river length using graph theory, 
                drainage density and shape factor. 

        Args:
            main_river_kwargs (dict, optional): 
                Additional arguments for the main river finding routine.
                Defaults to {}. Details in src.geomorphology.main_river routine
            slope_gdal_kwargs (dict, optional): 
                Additional arguments for the slope computing function.
                Defaults to {}.
            aspect_gdal_kwargs (dict, optional): 
                Additional arguments for the aspect computing function.
                Defaults to {}.

        Returns:
            pandas.DataFrame: Table with basin geomorphological properties
        """
        if self.params.shape != (1, 0):
            self.params = pd.DataFrame([], index=[self.fid])

        # Compute slope and aspect. Update dem property
        try:
            curve = self.compute_hypsometry()
            slope = self.compute_slope(**slope_gdal_kwargs)
            aspect = self.compute_aspect(**aspect_gdal_kwargs)
            self.dem = xr.merge([self.dem, slope.copy(), aspect.copy()])
            self.dem.attrs = {'standard_name': 'terrain model',
                              'hypsometry_x': [f'{i:.2f}' for i in curve.index],
                              'hypsometry_y': [f'{j:3f}' for j in curve.values]}
            # DEM derived params
            terrain_params = basin_terrain_params(self.fid, self.dem)
        except Exception as e:
            terrain_params = pd.DataFrame([], index=[self.fid])
            print('PostProcess DEM Error:', e)

        # Geographical parameters
        try:
            geo_params = basin_geographical_params(self.fid, self.wgeometry)
        except Exception as e:
            geo_params = pd.DataFrame([], index=[self.fid])
            print('Geographical Parameters Error:', e)

        # Flow derived params
        try:
            mainriver = main_river(self.rgeometry, **main_river_kwargs)
            mriverlen = self.rgeometry.loc[mainriver.index].length.sum()/1e3
            if mriverlen.item() != 0:
                mriverlen = mriverlen.item()
            else:
                mriverlen = np.nan
            self.params['mriverlen_km'] = mriverlen
            rhod = self.rgeometry.length.sum()/geo_params['area_km2']/1e3
            Kf = geo_params['area_km2']/(self.params['mriverlen_km']**2)
            self.params['rhod_1'] = rhod
            self.params['Kf_1'] = Kf
        except Exception as e:
            print('Flow derived properties Error:', e)

        # Auxiliary raster distributions
        try:
            if len(self.aux_rasters) != 0:
                counts = []
                for raster in self.aux_rasters:
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
            print('Auxiliary rasters Error:', e)

        self.params = pd.concat([geo_params,
                                 terrain_params,
                                 self.params], axis=1)
        self.params = pd.concat([self.params.T, counts.T],
                                keys=['geoparams', 'aux_rasters'])

        return self
