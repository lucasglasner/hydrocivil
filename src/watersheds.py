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

from src.geomorphology import *


class WaterShed:
    def __init__(self, fid, wgeometry, rgeometry, dem):
        self.fid = fid
        self.wgeometry = wgeometry
        self.rgeometry = rgeometry
        self.dem = dem.squeeze().to_dataset(name='elevation')
        self.geoparams = pd.DataFrame([], index=[self.fid])

        if not self.wgeometry.crs.is_projected:
            error = 'Watershed geometry must be in a projected (UTM) crs !'
            raise RuntimeError(error)
        if not self.rgeometry.crs.is_projected:
            error = 'River network geometry must be in a projected (UTM) crs !'
            raise RuntimeError(error)

    def compute_slope(self, DEMProcessing_kwargs={},
                      open_rasterio_kwargs={}):
        gdal.DEMProcessing('.tmp.slope.tif',
                           self.dem.elevation.encoding['source'],
                           'slope', computeEdges=True, slopeFormat='percent',
                           **DEMProcessing_kwargs)
        slope = rxr.open_rasterio('.tmp.slope.tif', **open_rasterio_kwargs)
        slope = slope.squeeze().to_dataset(name='slope')
        slope = slope.where(slope != -9999)/100
        return slope

    def compute_aspect(self, DEMProcessing_kwargs={},
                       open_rasterio_kwargs={}):
        gdal.DEMProcessing('.tmp.aspect.tif',
                           self.dem.elevation.encoding['source'],
                           'aspect', computeEdges=True, **DEMProcessing_kwargs)
        aspect = rxr.open_rasterio('.tmp.aspect.tif', **open_rasterio_kwargs)
        aspect = aspect.squeeze().to_dataset(name='aspect')
        aspect = aspect.where(aspect != -9999)
        return aspect

    def compute_geoparams(self, main_river_kwargs={}):
        # Geographical parameters
        geo_params = basin_geographical_params(self.fid, self.wgeometry)

        # DEM derived params
        terrain_params = basin_terrain_params(self.fid, self.dem)

        # Flow derived params
        mainriver = main_river(self.rgeometry, **main_river_kwargs)
        mriverlen = self.rgeometry.loc[mainriver.index].length.sum()/1e3
        self.geoparams['mriverlen_km'] = mriverlen.item()

        rhod = self.rgeometry.length.sum()/geo_params['area_km2']
        Kf = geo_params['area_km2']/(self.geoparams['mriverlen_km']**2)
        self.geoparams['rhod_1'] = rhod
        self.geoparams['Kf_1'] = Kf
        self.geoparams = pd.concat([geo_params,
                                    terrain_params,
                                    self.geoparams], axis=1).T
        return self.geoparams

    def compute(self, slope_gdal_kwargs={}, aspect_gdal_kwargs={}):

        # Get slope and aspect
        slope = self.compute_slope(**slope_gdal_kwargs)
        aspect = self.compute_aspect(**aspect_gdal_kwargs)
        self.dem = xr.merge([self.dem, slope, aspect])

        # Get geomorphological properties
        self.geoparams = self.compute_geoparams()

        # Clean tmp files
        os.remove('.tmp.aspect.tif')
        os.remove('.tmp.slope.tif')
        return self
