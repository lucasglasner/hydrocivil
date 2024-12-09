'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 2024-08-05 11:11:38
 Modified by: Lucas Glasner,
 Modified time: 2024-08-05 11:11:43
 Description: Main watershed classes
 Dependencies:
'''

import os
import warnings

import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr

from osgeo import gdal
from scipy.interpolate import interp1d

from .unithydrographs import SynthUnitHydro
from .geomorphology import get_main_river, concentration_time
from .geomorphology import basin_geographical_params, basin_terrain_params
from .misc import raster_distribution
from .global_vars import ABZONE_POLYGON
from .abstractions import cn_correction

# ---------------------------------------------------------------------------- #


class RiverBasin(object):
    """
    Watershed class used to compute geomorphological properties of basins, 
    unit hydrographs, flood hydrographs, terrain properties, among other 
    hydrological methods. 
    The class seeks to be a virtual representation of an hydrographic basin, 
    where given inputs of terrain, river network and land cover are used to 
    derive basin-wide properties and hydrological computations. 

    Examples:
        #### Compute geomorphometric parameters

        -> import geopandas as gpd
        -> import rioxarray as rxr
        -> dem = rxr.open_rasterio('path/to/dem', masked=True)
        -> cn  = rxr.open_rasterio('path/to/cn', masked=True)
        -> basin = gpd.read_file('/path/to/basinpolygon')
        -> rivers = gpd.read_file('/path/to/riversegments/')

        -> wshed = RiverBasin('mybasin', basin, rivers, dem, cn)
        -> wshed.compute_params()

        #### Use curve number corrected by a wet/dry condition
        -> wshed = RiverBasin('mybasin', basin, rivers, dem, cn, amc='wet')
        -> wshed.compute_params()

        #### Change a parameter by hand
        -> wshed.set_parameter('area', 1000)

        #### Check hypsometric curve
        -> curve = wshed.get_hypsometric_curve(bins='auto')

        #### Check fraction of area below 1400 meters
        -> fArea = wshed.area_below_height(1400)

        #### Access basin params as pandas Data Frame
        -> wshed.params

        #### Compute SCS unit hydrograph for rain pulses of 1 hour
        -> wshed.SynthUnitHydro(kind='SCS', timestep=1)

        #### Compute flood hydrograph with a series of rainfall
        -> whsed.UnitHydro.convolve(rainfall)
    """

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
        if not cn.rio.crs.is_projected:
            error = prj_error.format('Curve Number raster')
            raise RuntimeError(error)

    def __init__(self, fid, basin, rivers, dem, cn=None, amc='II'):
        """
        Drainage Basin class constructor

        Args:
            fid (str): Basin identifier
            basin (GeoDataFrame): Watershed polygon
            rivers (GeoDataFrame): River network segments
            dem (xarray.DataArray): Digital elevation model
            cn (xarray.DataArray): Curve Number raster.
                Defaults to None which leads to a full NaN curve number raster
            amc (str): Antecedent moisture condition. Defaults to 'II'. 
            Options: 'dry' or 'I',
                     'normal' or 'II'
                     'wet' or 'III'

        Raises:
            RuntimeError: If any of the given spatial data isnt in a projected
                cartographic projection.
        """
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
            self.cn = cn.rio.write_nodata(-9999).squeeze()
            self.cn = cn_correction(self.cn, amc=amc)
            self.cn_counts = pd.DataFrame([])
        else:
            self.cn = dem.squeeze()*np.nan

        # Properties
        self.params = pd.DataFrame([], index=[self.fid], dtype=object)
        self.hypsometric_curve = pd.Series([])

        # UnitHydrograph
        self.UnitHydro = None

        # Tests
        self.tests(self.basin, self.rivers, self.dem, self.cn)

    def __repr__(self) -> str:
        """
        What to show when invoking a RiverBasin object
        Returns:
            str: Some metadata
        """
        if type(self.UnitHydro) != type(None):
            uh_text = self.UnitHydro.method
        else:
            uh_text = None

        if self.params.shape != (1, 0):
            param_text = str(self.params).replace(self.fid, '')
        else:
            param_text = None
        text = f'RiverBasin: {self.fid}\nUnitHydro: {uh_text}\n'
        text = text+f'Parameters: {param_text}'
        return text

    def set_parameter(self, index, value):
        """
        Simple function to add or fix a parameter to the basin parameters table

        Args:
            index (str): parameter name/id or what to put in the table index
            value (object): value of the new parameter
        """
        self.params.loc[index, :] = value
        return self

    def process_gdaldem(self, varname, overwrite=True,
                        open_rasterio_kwargs={}, **kwargs):
        """
        Accessor to gdaldem command line utility.

        Args:
            open_rasterio_kwargs (dict, optional):
                Arguments for rioxarray open rasterio function. Defaults to {}.

        Returns:
            xarray.DataArray: DEM derived property
        """
        ipath = os.path.abspath(self.dem.elevation.encoding['source'])
        fname = os.path.basename(self.dem.elevation.encoding['source'])
        oname = os.path.join(os.path.dirname(ipath), f'{varname}_{fname}')
        if os.path.isfile(oname) and overwrite:
            field = rxr.open_rasterio(oname, **open_rasterio_kwargs)
            field = field.squeeze().to_dataset(name=varname)
        else:
            gdal.DEMProcessing(oname, ipath, varname, **kwargs)
            field = rxr.open_rasterio(oname, **open_rasterio_kwargs)
            field = field.squeeze().to_dataset(name=varname)
        return field

    def get_hypsometric_curve(self, bins='auto', **kwargs):
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
            self.get_hypsometric_curve(**kwargs)
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
            curve = self.get_hypsometric_curve()
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

    def process_river_network(self):
        """
        Compute river network properties

        Returns:
            self: updated class
        """
        try:
            mainriver = get_main_river(self.rivers)
            self.rivers_main = mainriver
            mriverlen = self.rivers_main.length.sum()/1e3
            if mriverlen.item() != 0:
                mriverlen = mriverlen.item()
            else:
                mriverlen = np.nan
            self.params['mriverlen'] = mriverlen
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
                counts = counts.reset_index().rename({self.fid: 'weights'},
                                                     axis=1)
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

    def compute_params(self, dem_kwargs={}, geography_kwargs={},
                       river_network_kwargs={}):
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
            dem_kwargs (dict, optional): 
                Additional arguments for the terrain preprocessing function.
                Defaults to {}.
            geography_kwargs (dict, optional):
                Additional arguments for the geography preprocessing routine.
                Defauts to {}.
            river_network_kwargs (dict, optional): 
                Additional arguments for the main river finding routine.
                Defaults to {}. Details in src.geomorphology.main_river routine
        Returns:
            self: updated class
        """
        if self.params.shape != (1, 0):
            self.params = pd.DataFrame([], index=[self.fid])

        # Geographical parameters
        self.process_geography(**geography_kwargs)

        # Compute slope and aspect. Update dem property
        self.process_dem(masked=True, **dem_kwargs)

        # Flow derived params
        self.process_river_network(**river_network_kwargs)

        # Curve number process
        self.cn_counts = self.process_raster_counts(self.cn)
        self.params['curvenumber'] = self.cn.mean().item()
        self.params = self.params.T.astype(object)

        # self.params = pd.concat([self.params.T, cn_counts], axis=0)

        return self

    def SynthUnitHydro(self, method, **kwargs):
        """
        Synthetic Unit Hygrograph class accessor

        Args:
            method (str): Type of synthetic unit hydrograph to use. 
                Options: 'SCS', 'Gray', 'Arteaga&Benitez', 
            timestep (float): unit hydrograph timestep. 
        Returns:
            self: updated class
        """
        if method == 'Arteaga&Benitez':
            centroid = self.basin.centroid.to_crs('epsg:4326').loc[0]
            mask = centroid.within(ABZONE_POLYGON.geometry)
            if mask.sum() == 0:
                raise RuntimeError(
                    f'No valid {method} zone for {self.fid} basin')
            else:
                self.params.loc['zone_AB'] = ABZONE_POLYGON[mask].zone.item()
        SUH = SynthUnitHydro(self.params[self.fid], method, **kwargs).compute()
        self.UnitHydro = SUH
        return self

    def plot(self, basin_kwargs={}, rivers_kwargs={}, rivers_main_kwargs={}):
        """
        Simple plot function for the basin taking account polygon and rivers

        Args:
            basin_kwargs (dict, optional): Arguments for the basin.
                Defaults to {}.
            rivers_kwargs (dict, optional): Arguments for the rivers.
                Defaults to {}.
            rivers_main_kwargs (dict, optional): Arguments for the main rivers.
                Defaults to {}.

        Returns:
            matplotlib axes instance
        """
        plot_basin = self.basin.plot(color='silver', edgecolor='k',
                                     **basin_kwargs)
        plot_basin.axes.set_title(self.fid, loc='left')
        self.rivers.plot(ax=plot_basin.axes, **rivers_kwargs)
        if len(self.rivers_main) != 0:
            self.rivers_main.plot(ax=plot_basin.axes, color='tab:red',
                                  **rivers_main_kwargs)
        return plot_basin.axes


class RiverReach(object):
    def tests(self, rivers, dem):
        """
        Args:
            rivers (GeoDataFrame): River network lines
            dem (xarray.DataArray): Digital elevation model raster
        Raises:
            RuntimeError: If any dataset isnt in a projected (UTM) crs.
        """
        prj_error = '{} must be in a projected (UTM) crs !'
        if not rivers.crs.is_projected:
            error = prj_error.format('Rivers geometry')
            raise RuntimeError(error)
        if not dem.rio.crs.is_projected:
            error = prj_error.format('DEM raster')
            raise RuntimeError(error)

    def __init__(self, fid, dem, rivers):
        """
        River reach/channel class constructor

        Args:
            fid (str): Basin identifier
            rivers (GeoDataFrame): River network segments
            dem (xarray.DataArray): Digital elevation model

        Raises:
            RuntimeError: If any of the given spatial data isnt in a projected
                cartographic projection.
        """
        self.tests(rivers, dem)
        # ID
        self.fid = fid

        # Vectors
        self.rivers = rivers
        self.rivers_main = pd.Series([])

        # Terrain
        self.dem = dem.rio.write_nodata(-9999).squeeze()
        self.dem = self.dem.to_dataset(name='elevation')
        self.dem.encoding = dem.encoding


class Reservoir(object):
    def __init__(self):
        pass
