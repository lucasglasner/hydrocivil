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
from src.misc import get_psep, raster_distribution


class DrainageBasin:
    def __init__(self, fid, wgeometry, rgeometry, dem, lulc=[]):
        """
        Drainage Basin class constructor

        Args:
            fid (str): Basin identifier
            wgeometry (GeoDataFrame): Watershed/Basin polygon
            rgeometry (GeoDataFrame): RiverNetwork segments
            dem (xarray.DataArray): Digital elevation model
            lulc (list): list of additional xarray.DataArray categorical
                         land cover rasters.

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
        self.lulc = lulc

    def __repr__(self) -> str:
        """
        What to show when invoking a DrainageBasin object
        Returns:
            str: Some metadata
        """
        text = f'DrainageBasin\nFID: {self.fid}\n{self.params.T}'
        return text

    def compute_gdaldem(self, varname, open_rasterio_kwargs={}, **kwargs):
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

    def compute_hypsometry(self, bins='auto', **kwargs):
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
            print('Computing hypsometric curve ...')
            self.compute_hypsometry(**kwargs)
        curve = self.hypsometric_curve
        if height < curve.index.min():
            return 0
        if height > curve.index.max():
            return 1
        interp_func = interp1d(curve.index.values, curve.values)
        return interp_func(height).item()

    def process_dem(self, **kwargs):
        """
        Compute hypsometric curve, slope and aspect. Then compute DEM
        derived propeties for the basin and save in the params dataframe.

        Returns:
            self: updated class
        """
        try:
            curve = self.compute_hypsometry()
            slope = self.compute_gdaldem('slope',
                                         computeEdges=True,
                                         slopeFormat='percent',
                                         open_rasterio_kwargs={**kwargs})
            aspect = self.compute_gdaldem('aspect',
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
            print('PostProcess DEM Error:', e, self.fid)
        self.params = pd.concat([self.params, terrain_params], axis=1)
        return self

    def process_geography(self):
        """
        Compute geographical parameters of the basin

        Returns:
            self: updated class
        """
        try:
            geo_params = basin_geographical_params(self.fid, self.wgeometry)
        except Exception as e:
            geo_params = pd.DataFrame([], index=[self.fid])
            print('Geographical Parameters Error:', e, self.fid)
        self.params = pd.concat([self.params, geo_params], axis=1)
        return self

    def process_river_network(self, **kwargs):
        """
        Compute river network properties

        Returns:
            self: updated class
        """
        try:
            mainriver = main_river(self.rgeometry, **kwargs)
            mriverlen = self.rgeometry.loc[mainriver.index].length.sum()/1e3
            if mriverlen.item() != 0:
                mriverlen = mriverlen.item()
            else:
                mriverlen = np.nan
            self.params['mriverlen_km'] = mriverlen
            rhod = self.rgeometry.length.sum()/self.params['area_km2']/1e3
            Kf = self.params['area_km2']/(self.params['mriverlen_km']**2)
            self.params['rhod_1'] = rhod
            self.params['Kf_1'] = Kf
        except Exception as e:
            print('Flow derived properties Error:', e, self.fid)
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
            print('LULC rasters Error:', e, self.fid)
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
                Detalils in src.geomorphology.basin_terrain_params
            3) Flow derived properties: Main river length using graph theory, 
                drainage density and shape factor. 

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

        # Compute slope and aspect. Update dem property
        self.process_dem(masked=True, **gdal_kwargs)

        # Geographical parameters
        self.process_geography()

        # Flow derived params
        self.process_river_network(**main_river_kwargs)

        # Auxiliary raster distributions
        counts = self.process_lulc()

        self.params = pd.concat([self.params.T, counts.T],
                                keys=['geoparams', 'lulc'])

        return self
