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


class DrainageBasin:
    def __init__(self, fid, wgeometry, rgeometry, dem):
        """
        Drainage Basin class constructor

        Args:
            fid (str): Basin identifier
            wgeometry (GeoDataFrame): Watershed/Basin polygon
            rgeometry (GeoDataFrame): RiverNetwork segments
            dem (xarray.DataArray): Digital elevation model

        Raises:
            RuntimeError: If any of the given spatial data isnt in a projected
                cartographic projection.
        """
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

    def __repr__(self) -> str:
        """
        What to show when invoking a DrainageBasin object
        Returns:
            str: Some metadata
        """
        return f'DrainageBasin\nFID: {self.fid}\n{self.geoparams}'

    def compute_slope(self, DEMProcessing_kwargs={},
                      open_rasterio_kwargs={}):
        """
        Computes slope raster from DEM using standard gdal routines

        Args:
            DEMProcessing_kwargs (dict, optional):
                Arguments for gdal DEMProcessing algorithm. Defaults to {}.
            open_rasterio_kwargs (dict, optional):
                Arguments for rioxarray open rasterio function. Defaults to {}.

        Returns:
            xarray.DataArray: DEM derived slopes in m/m
        """
        gdal.DEMProcessing('.tmp.slope.tif',
                           self.dem.elevation.encoding['source'],
                           'slope', computeEdges=True, slopeFormat='percent',
                           **DEMProcessing_kwargs)
        slope = rxr.open_rasterio('.tmp.slope.tif', masked=True,
                                  **open_rasterio_kwargs)
        slope = slope.squeeze().to_dataset(name='slope')
        slope = slope.where(slope != -9999)/100
        return slope

    def compute_aspect(self, DEMProcessing_kwargs={},
                       open_rasterio_kwargs={}):
        """
        Computes aspect raster from DEM using standard gdal routines

        Args:
            DEMProcessing_kwargs (dict, optional):
                Arguments for gdal DEMProcessing algorithm. Defaults to {}.
            open_rasterio_kwargs (dict, optional):
                Arguments for rioxarray open rasterio function. Defaults to {}.

        Returns:
            xarray.DataArray: DEM derived aspect in degrees
        """
        gdal.DEMProcessing('.tmp.aspect.tif',
                           self.dem.elevation.encoding['source'],
                           'aspect', computeEdges=True, **DEMProcessing_kwargs)
        aspect = rxr.open_rasterio('.tmp.aspect.tif', **open_rasterio_kwargs)
        aspect = aspect.squeeze().to_dataset(name='aspect')
        aspect = aspect.where(aspect != -9999)
        return aspect

    def compute_geoparams(self, main_river_kwargs={}, slope_gdal_kwargs={},
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
        # Compute slope and aspect. Update dem property
        slope = self.compute_slope(**slope_gdal_kwargs)
        aspect = self.compute_aspect(**aspect_gdal_kwargs)
        self.dem = xr.merge([self.dem, slope, aspect])

        # Geographical parameters
        geo_params = basin_geographical_params(self.fid, self.wgeometry)

        # DEM derived params
        terrain_params = basin_terrain_params(self.fid, self.dem)

        # Flow derived params
        mainriver = main_river(self.rgeometry, **main_river_kwargs)
        mriverlen = self.rgeometry.loc[mainriver.index].length.sum()/1e3
        self.geoparams['mriverlen_km'] = mriverlen.item()

        rhod = self.rgeometry.length.sum()/geo_params['area_km2']/1e3
        Kf = geo_params['area_km2']/(self.geoparams['mriverlen_km']**2)
        self.geoparams['rhod_1'] = rhod
        self.geoparams['Kf_1'] = Kf
        self.geoparams = pd.concat([geo_params,
                                    terrain_params,
                                    self.geoparams], axis=1).T
        os.remove('.tmp.aspect.tif')
        os.remove('.tmp.slope.tif')

        return self
