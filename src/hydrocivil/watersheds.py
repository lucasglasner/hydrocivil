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
import copy as pycopy

from osgeo import gdal, gdal_array
from scipy.interpolate import interp1d

from .misc import raster_distribution, polygonize, gdal2xarray, xarray2gdal
from .unithydrographs import LumpedUnitHydrograph as SUH
from .geomorphology import get_main_river, basin_outlet
from .geomorphology import basin_geographical_params, basin_terrain_params
from .global_vars import CHILE_UH_LINSLEYPOLYGONS, CHILE_UH_GRAYPOLYGONS
from .global_vars import GDAL_EXCEPTIONS
from .abstractions import cn_correction
from .abstractions import SCS_EffectiveRainfall, SCS_EquivalentCurveNumber

if GDAL_EXCEPTIONS:
    gdal.UseExceptions()
else:
    gdal.DontUseExceptions()
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
        + Compute geomorphometric parameters

        -> import geopandas as gpd
        -> import rioxarray as rxr
        -> dem = rxr.open_rasterio('path/to/dem', masked=True)
        -> cn  = rxr.open_rasterio('path/to/cn', masked=True)
        -> basin = gpd.read_file('/path/to/basinpolygon')
        -> rivers = gpd.read_file('/path/to/riversegments/')

        -> wshed = RiverBasin('mybasin', basin, rivers, dem, cn)
        -> wshed.compute_params()

        + Use curve number corrected by a wet/dry condition
        -> wshed = RiverBasin('mybasin', basin, rivers, dem, cn, amc='wet')
        -> wshed.compute_params()

        + Change a parameter by hand
        -> wshed.set_parameter('area', 1000)

        + Check hypsometric curve
        -> curve = wshed.get_hypsometric_curve(bins='auto')

        + Check fraction of area below 1400 meters
        -> fArea = wshed.area_below_height(1400)

        + Get relationship of curve number vs precipitation due to basin land
        + cover heterogeneities. 
        -> cn_curve = wshed.get_equivalent_curvenumber()

        + Access basin params as pandas Data Frame
        -> wshed.params

        + Compute SCS unit hydrograph for rain pulses of 1 hour
        -> wshed.SynthUnitHydro(method='SCS', timestep=1)

        + Compute flood hydrograph with a series of rainfall
        -> whsed.UnitHydro.convolve(rainfall)
    """

    def _tests(self, basin, rivers, dem, cn):
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
        self.basin = basin.copy()
        self.rivers = rivers.copy()
        self.rivers_main = pd.Series([])

        # Terrain
        self.dem = dem.rio.write_nodata(-9999).squeeze().copy()
        self.dem = self.dem.to_dataset(name='elevation')
        self.dem.encoding = dem.encoding

        # Curve Number
        self.amc = amc
        if type(cn) != type(None):
            self.cn = cn.rio.write_nodata(-9999).squeeze().copy()
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
        self._tests(self.basin, self.rivers, self.dem, self.cn)

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

    def copy(self):
        """
        Create a deep copy of the class itself
        """
        return pycopy.deepcopy(self)

    def _process_gdaldem(self, varname, **kwargs):
        """
        Accessor to gdaldem command line utility.

        Args:


        Returns:
            xarray.DataArray: DEM derived property
        """
        dem_xr = self.dem.elevation
        dem_gdal = xarray2gdal(dem_xr)

        # Create in-memory output GDAL dataset
        dtype = gdal_array.NumericTypeCodeToGDALTypeCode(dem_xr.dtype)
        mem_driver = gdal.GetDriverByName('MEM')
        out_ds = mem_driver.Create('', dem_xr.sizes['x'], dem_xr.sizes['y'], 1,
                                   dtype)
        out_ds.SetGeoTransform(dem_xr.rio.transform().to_gdal())
        out_ds.SetProjection(dem_xr.rio.crs.to_wkt())

        # Process DEM using gdal.DEMProcessing
        out_ds = gdal.DEMProcessing(out_ds.GetDescription(), dem_gdal, varname,
                                    format='MEM', computeEdges=True, **kwargs)
        out_ds = gdal2xarray(out_ds).to_dataset(name=varname)
        out_ds.coords['y'] = dem_xr.coords['y']
        out_ds.coords['x'] = dem_xr.coords['x']
        return out_ds

    def _processgeography(self, n=3, **kwargs):
        """
        Compute geographical parameters of the basin

        Args:
            n (int, optional): Number of DEM pixels to consider for the
                elevation boundary. Defaults to 3.
            **kwargs are given to basin_geographical_params function.

        Returns:
            self: updated class
        """
        try:
            cond1 = 'outlet_x' not in self.basin.columns
            cond2 = 'outlet_y' not in self.basin.columns
            if cond1 or cond2:
                outlet_y, outlet_x = self.get_basin_outlet(n=n)

            geo_params = basin_geographical_params(self.fid, self.basin,
                                                   **kwargs)
        except Exception as e:
            geo_params = pd.DataFrame([], index=[self.fid])
            warnings.warn('Geographical Parameters Error:', e, self.fid)
        self.params = pd.concat([self.params, geo_params], axis=1)
        return self

    def _processdem(self, gdalpreprocess=True):
        """
        Compute hypsometric curve, slope and aspect. Then compute DEM
        derived propeties for the basin and save in the params dataframe.

        Returns:
            self: updated class
        """
        try:
            if gdalpreprocess:
                curve = self.get_hypsometric_curve()
                slope = self._process_gdaldem('slope', slopeFormat='percent')
                aspect = self._process_gdaldem('aspect')

                slope = slope.where(slope != -9999)
                aspect = aspect.where(aspect != -9999)

                self.dem = xr.merge([self.dem.elevation, slope/100, aspect])
                self.dem.attrs = {
                    'standard_name': 'terrain model',
                    'hypsometry_x': [f'{i:.2f}' for i in curve.index],
                    'hypsometry_y': [f'{j:3f}' for j in curve.values]
                }
            # DEM derived params
            terrain_params = basin_terrain_params(self.fid, self.dem)
        except Exception as e:
            terrain_params = pd.DataFrame([], index=[self.fid])
            warnings.warn('PostProcess DEM Error:', e, self.fid)
        self.params = pd.concat([self.params, terrain_params], axis=1)
        return self

    def _processrivers(self):
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

    def _processrastercounts(self, raster, output_type=1):
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

    def set_parameter(self, index, value):
        """
        Simple function to add or fix a parameter to the basin parameters table

        Args:
            index (str): parameter name/id or what to put in the table index
            value (object): value of the new parameter
        """
        self.params.loc[index, :] = value
        return self

    def get_basin_outlet(self, n=3):
        """
        This function computes the basin outlet point defined as the
        point of minimum elevation along the basin boundary.

        Args:
            n (int, optional): Number of DEM pixels to consider for the
                elevation boundary. Defaults to 3.

        Returns:
            outlet_y, outlet_x (tuple): Tuple with defined outlet y and x
                coordinates.
        """
        outlet_y, outlet_x = basin_outlet(self.basin, self.dem.elevation, n=n)
        self.basin['outlet_x'] = outlet_x
        self.basin['outlet_y'] = outlet_y
        return (outlet_y, outlet_x)

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

    def get_equivalent_curvenumber(self, pr_range=(1, 1000), **kwargs):
        """
        This routine calculates the dependence of the watershed curve number
        on precipitation due to land cover heterogeneities. 
        Args:
            pr_range (tuple): Minimum and maximum possible precipitations
            **kwargs are given to SCS_EffectiveRainfall routine.

        Returns:
            (array_like): Basin curve number as a function of precipitation (mm)
        """
        # Precipitation range
        pr = np.linspace(pr_range[0], pr_range[1], 1000)
        pr = np.expand_dims(pr, axis=-1)

        # Curve number counts
        cn_counts = self._processrastercounts(self.cn)
        weights, cn_values = cn_counts['weights'].values, cn_counts['cn'].values
        cn_values = np.expand_dims(cn_values, axis=-1)

        # Broadcast curve number and pr arrays
        broad = np.broadcast_arrays(cn_values, pr.T)

        # Get effective precipitation
        pr_eff = SCS_EffectiveRainfall(pr=broad[1], cn=broad[0], **kwargs)
        pr_eff = (pr_eff.T * weights).sum(axis=-1)

        # Compute equivalent curve number for hetergeneous basin
        curve = SCS_EquivalentCurveNumber(pr[:, 0], pr_eff, **kwargs)
        curve = pd.Series(curve, index=pr[:, 0])
        curve = curve.sort_index()
        self.cn_equivalent = curve
        return curve

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
        self._processgeography(**geography_kwargs)

        # Compute slope and aspect. Update dem property
        self._processdem(**dem_kwargs)

        # Flow derived params
        self._processrivers(**river_network_kwargs)

        # Curve number process
        # self.cn_counts = self._processrastercounts(self.cn)
        self.params['curvenumber'] = self.cn.mean().item()
        self.params = self.params.T.astype(object)

        # self.params = pd.concat([self.params.T, cn_counts], axis=0)

        return self

    def clip(self, polygon, **kwargs):
        """
        Clip watershed data into the given polygon. Update geomorphometric
        parameters and inner class data. 
        Args:
            polygon (_type_): Polygon (must be in the same crs)

        Returns:
            New RiverBasin object with clipped data
        """
        polygon = polygon.dissolve()
        nbasin = self.basin.copy().clip(polygon)
        nrivers = self.rivers.copy().clip(polygon)
        ndem = self.dem.copy().rio.clip(polygon.geometry)
        ncn = self.cn.copy().rio.clip(polygon.geometry)

        ndem = ndem.where(ndem != -9999)
        ncn = ncn.where(ncn != -9999)

        nwshed = self.copy()
        nwshed.basin = nbasin
        nwshed.rivers = nrivers
        nwshed.dem = ndem
        nwshed.cn = ncn

        if 'dem_kwargs' in kwargs.keys():
            kwargs['dem_kwargs']['preprocess'] = False
        else:
            kwargs['dem_kwargs'] = {'preprocess': False}
        nwshed.compute_params(**kwargs)
        return nwshed

    def update_snowline(self, snowline, polygonize_kwargs={}, **kwargs):
        """
        From a given snowline height, this routine updates the basin object to
        the target basin pluvial area, changing the basin polygon, data and
        geomorphometric properties. 

        Args:
            snowline (float): Height of the snowline (same units as input dem)
            polygonize_kwargs (dict, optional): Aditional arguments to
                polygonize function. Defaults to {}.
        Raises:
            ValueError: If snowline is outside DEM elevation range
        Returns:
            updated class
        """
        if not isinstance(snowline, (int, float)):
            raise TypeError("Snowline must be numeric")
        min_elev = self.dem.elevation.min().item()
        if snowline < min_elev:
            raise ValueError(f"Snowline {snowline} below hmin {min_elev}")
        nshp = polygonize(self.dem.elevation < snowline, **polygonize_kwargs)
        return self.clip(nshp, **kwargs)

    def SynthUnitHydro(self, method, ChileParams=False, **kwargs):
        """
        Synthetic Unit Hygrograph class accessor

        Args:
            method (str): Type of synthetic unit hydrograph to use. 
                Options: 'SCS', 'Gray', 'Arteaga&Benitez', 
            ChileParams (bool): Whether to use Chile-specific parameters
            **kwargs are given to the synthetic unit hydrograph routine.
        Returns:
            (RiverBasin): Updated RiverBasin instance
        """
        if (method == 'Linsley') and ChileParams:
            poly = CHILE_UH_LINSLEYPOLYGONS
            centroid = self.basin.centroid.to_crs('epsg:4326').loc[0]
            mask = centroid.within(poly.geometry)
            if mask.sum() == 0:
                text = f'Basin {self.fid} is outside the geographical limits'
                text = text+f' allowed by the Chilean {method} method.'
                raise RuntimeError(text)
            else:
                DGAChile_LinsleyZone = poly[mask].zone.item()
                self.params.loc['DGAChile_LinsleyZone'] = DGAChile_LinsleyZone
                uh = SUH(method, self.params[self.fid])
                uh = uh.compute(DGAChileParams=True,
                                DGAChileZone=DGAChile_LinsleyZone, **kwargs)
        elif (method == 'Gray') and ChileParams:
            poly = CHILE_UH_GRAYPOLYGONS
            centroid = self.basin.centroid.to_crs('epsg:4326').loc[0]
            mask = centroid.within(poly.geometry)
            if mask.sum() == 0:
                text = f'Basin {self.fid} is outside the geographical limits'
                text = text+f' allowed by the Chilean {method} method.'
                raise RuntimeError(text)
            else:
                uh = SUH(method, self.params[self.fid])
                uh = uh.compute(DGAChileParams=True, **kwargs)
        else:
            uh = SUH(method, self.params[self.fid])
            uh = uh.compute(**kwargs)
        self.UnitHydro = uh
        return self

    def plot(self, outlet_kwargs={'ec': 'k', 'color': 'tab:red', 'zorder': 10},
             basin_kwargs={'color': 'silver', 'edgecolor': 'k'},
             rivers_kwargs={'color': 'tab:blue'},
             rivers_main_kwargs={'color': 'tab:red'}):
        """
        Simple plot function for the basin taking account polygon and rivers

        Args:
            outlet_kwargs (dict, optional): Arguments for the basin outlet.
                Defaults to {'ec': 'k', 'color': 'tab:red', 'zorder': 10}.
            basin_kwargs (dict, optional): Arguments for the basin.
                Defaults to {'color': 'silver', 'edgecolor': 'k'}.
            rivers_kwargs (dict, optional): Arguments for the rivers.
                Defaults to {'color': 'tab:blue'}.
            rivers_main_kwargs (dict, optional): Arguments for the main rivers.
                Defaults to {'color': 'tab:red'}.

        Returns:
            matplotlib axes instance
        """
        plot_basin = self.basin.plot(**basin_kwargs)
        plot_basin.axes.set_title(self.fid, loc='left')
        self.rivers.plot(ax=plot_basin.axes, **rivers_kwargs)
        plot_basin.axes.scatter(self.basin['outlet_x'], self.basin['outlet_y'],
                                **outlet_kwargs)
        if len(self.rivers_main) != 0:
            self.rivers_main.plot(ax=plot_basin.axes, **rivers_main_kwargs)
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
