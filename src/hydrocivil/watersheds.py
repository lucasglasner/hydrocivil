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
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import copy as pycopy
import matplotlib

from typing import Union, Any, Optional, Type, Tuple
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

    def _tests(self,
               basin: Union[gpd.GeoSeries, gpd.GeoDataFrame],
               rivers: Union[gpd.GeoSeries, gpd.GeoDataFrame],
               dem: xr.DataArray,
               cn: xr.DataArray) -> None:
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

    def __init__(self, fid: Union[str, int, float],
                 basin: Union[gpd.GeoSeries, gpd.GeoDataFrame],
                 rivers: Union[gpd.GeoSeries, gpd.GeoDataFrame],
                 dem: xr.DataArray,
                 cn: Union[xr.DataArray, None] = None,
                 amc: Optional[str] = 'II') -> None:
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

    def copy(self) -> Type['RiverBasin']:
        """
        Create a deep copy of the class itself
        """
        return pycopy.deepcopy(self)

    def _process_gdaldem(self, varname: str, **kwargs: Any) -> xr.Dataset:
        """
        Processes a Digital Elevation Model (DEM) using the GDAL DEMProcessing
        utility. This method utilizes the GDAL DEMProcessing command line
        utility to derive various properties from a DEM. The output is returned
        as an xarray Dataset.

        Args:
            varname (str): The name of the DEM derived property to compute.
            **kwargs (Any): Additional keyword arguments to pass to the GDAL
                DEMProcessing function.

        Returns:
            xr.Dataset: An xarray Dataset containing the DEM derived property.
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

    def _processgeography(self, n: int = 3,
                          **kwargs: Any) -> Type['RiverBasin']:
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

    def _processdem(self, gdalpreprocess: bool = True) -> Type['RiverBasin']:
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

    def _processrivers(self) -> Type['RiverBasin']:
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

    def _processrastercounts(self, raster: xr.DataArray, output_type: int = 1
                             ) -> pd.DataFrame:
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

    def set_parameter(self, index: str, value: Any) -> Type['RiverBasin']:
        """
        Simple function to add or fix a parameter to the basin parameters table

        Args:
            index (str): parameter name/id or what to put in the table index
            value (Any): value of the new parameter
        """
        self.params.loc[index, :] = value
        return self

    def get_basin_outlet(self, n: int = 3) -> Tuple[np.ndarray, np.ndarray]:
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

    def get_hypsometric_curve(self, bins: Union[str, int, float] = 'auto',
                              **kwargs: Any) -> pd.Series:
        """
        Compute the hypsometric curve of the basin based on terrain elevation
        data. The hypsometric curve represents the distribution of elevation
        within the basin, expressed as the fraction of the total area that lies
        below a given elevation.
        Args:
            bins (str|int|float, optional): The method or number of
                bins to use for the elevation distribution. Default is 'auto'.
            **kwargs (Any): Additional keyword arguments to pass to the
                raster_distribution function.

        Returns:
            pandas.Series: A pandas Series representing the hypsometric curve,
                where the index corresponds to elevation bins and the values
                represent the cumulative fraction of the area below each
                elevation.
        """
        curve = raster_distribution(self.dem.elevation, bins=bins, **kwargs)
        self.hypsometric_curve = curve.cumsum()
        return curve

    def area_below_height(self, height: Union[int, float], **kwargs: Any
                          ) -> float:
        """
        With the hypsometric curve compute the fraction of area below
        a certain height.

        Args:
            height (int|float): elevation value
            **kwargs (Any): Additional keyword arguments to pass to the
                raster_distribution function.

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

    def get_equivalent_curvenumber(self,
                                   pr_range: Tuple[float, float] = (1., 1000.),
                                   **kwargs: Any) -> pd.Series:
        """
        Calculate the dependence of the watershed curve number on precipitation
        due to land cover heterogeneities.

        This routine computes the equivalent curve number for a heterogeneous
        basin as a function of precipitation. It takes into account the
        distribution of curve numbers within the basin and the corresponding
        effective rainfall for a range of precipitation values.

        Args:
            pr_range (tuple): Minimum and maximum possible precipitation (mm).
            **kwargs: Additional keyword arguments to pass to the
                SCS_EffectiveRainfall and SCS_EquivalentCurveNumber routine.

        Returns:
            pd.Series: A pandas Series representing the equivalent curve number
                as a function of precipitation, where the index corresponds to
                precipitation values and the values represent the equivalent
                curve number.
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

    def compute_params(self, dem_kwargs: dict = {},
                       geography_kwargs: dict = {},
                       river_network_kwargs: dict = {}) -> Type['RiverBasin']:
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
        self.params['curvenumber'] = self.cn.mean().item()
        self.params = self.params.T.astype(object)

        return self

    def clip(self, polygon: Union[gpd.GeoSeries, gpd.GeoDataFrame],
             **kwargs: Any) -> Type['RiverBasin']:
        """
        Clip watershed data to a specified polygon boundary and create a new
        RiverBasin object. This method creates a new RiverBasin instance with
        all data (basin boundary, rivers, DEM, etc) clipped to the given
        polygon boundary. It also recomputes all geomorphometric parameters for
        the clipped area.

        Args:
            polygon (Union[gpd.GeoSeries, gpd.GeoDataFrame]): Polygon defining
                the clip boundary. Must be in the same coordinate reference
                system (CRS) as the watershed data.
            **kwargs (Any): Additional keyword arguments to pass to
                self.compute_params() method.
            dem_kwargs (dict, optional): Dictionary of keyword arguments for
                DEM processing. If provided, preprocess will be set to False.
        Returns:
            self: A new RiverBasin object containing the clipped data and
                updated parameters.
        Notes:
            - The input polygon will be dissolved to ensure a single boundary
            - No-data values (-9999) are filtered out from DEM and CN rasters
            - All geomorphometric parameters are recomputed for the clipped
              area
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

    def update_snowline(self, snowline: Union[int, float],
                        polygonize_kwargs: dict = {},
                        **kwargs: Any) -> Type['RiverBasin']:
        """
        Updates the RiverBasin object to represent only the pluvial (rain-fed) 
        portion of the watershed below a given snowline elevation.

        This method clips the basin to areas below the specified snowline 
        elevation threshold. The resulting watershed represents only the
        portion of the basin that receives precipitation as rainfall rather
        than snow. All watershed properties (area, rivers, DEM, etc.) are
        updated accordingly.

        Args:
            snowline (int|float): Elevation threshold in same units as DEM that 
                defines the rain/snow transition zone
            polygonize_kwargs (dict, optional): Additional keyword arguments 
                passed to the polygonize function. Defaults to {}.
            **kwargs: Additional keyword arguments passed to the clip method

        Raises:
            TypeError: If snowline argument is not numeric
            ValueError: If snowline elevation is below minimum basin elevation

        Returns:
            RiverBasin: A new RiverBasin object containing only the pluvial
                portion of the original watershed below the snowline
        """
        if not isinstance(snowline, (int, float)):
            raise TypeError("Snowline must be numeric")
        min_elev = self.dem.elevation.min().item()
        if snowline < min_elev:
            raise ValueError(f"Snowline {snowline} below hmin {min_elev}")
        nshp = polygonize(self.dem.elevation < snowline, **polygonize_kwargs)
        return self.clip(nshp, **kwargs)

    def SynthUnitHydro(self, method: str, ChileParams: bool = False,
                       **kwargs: Any) -> Type['RiverBasin']:
        """
        Compute synthetic unit hydrograph for the basin.

        This method creates and computes a synthetic unit hydrograph based
        on basin parameters. For Chilean watersheds, special regional parameters
        can be used if ChileParams = True.

        Args:
            method (str): Type of synthetic unit hydrograph to use.
                Options: 
                    - 'SCS': SCS dimensionless unit hydrograph
                    - 'Gray': Gray's method
                    - 'Linsley': Linsley method
            ChileParams (bool): Whether to use Chile-specific regional
                parameters. Only valid for 'Gray' and 'Linsley' methods.
                Defaults to False.
            **kwargs: Additional arguments passed to the unit hydrograph
                computation method.

        Returns:
            RiverBasin: Updated instance with computed unit hydrograph stored in 
                UnitHydro attribute.

        Raises:
            RuntimeError: If using Chilean parameters and basin centroid lies outside 
                valid geographical regions.
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

    def plot(self,
             outlet_kwargs: dict = {'ec': 'k',
                                    'color': 'tab:red', 'zorder': 2},
             basin_kwargs: dict = {'color': 'silver', 'edgecolor': 'k'},
             rivers_kwargs: dict = {'color': 'tab:blue', 'alpha': 0.5},
             rivers_main_kwargs: dict = {'color': 'tab:red'}
             ) -> matplotlib.axes.Axes:
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


# class RiverReach(object):
#     def tests(self, rivers, dem):
#         """
#         Args:
#             rivers (GeoDataFrame): River network lines
#             dem (xarray.DataArray): Digital elevation model raster
#         Raises:
#             RuntimeError: If any dataset isnt in a projected (UTM) crs.
#         """
#         prj_error = '{} must be in a projected (UTM) crs !'
#         if not rivers.crs.is_projected:
#             error = prj_error.format('Rivers geometry')
#             raise RuntimeError(error)
#         if not dem.rio.crs.is_projected:
#             error = prj_error.format('DEM raster')
#             raise RuntimeError(error)

#     def __init__(self, fid, dem, rivers):
#         """
#         River reach/channel class constructor

#         Args:
#             fid (str): Basin identifier
#             rivers (GeoDataFrame): River network segments
#             dem (xarray.DataArray): Digital elevation model

#         Raises:
#             RuntimeError: If any of the given spatial data isnt in a projected
#                 cartographic projection.
#         """
#         self.tests(rivers, dem)
#         # ID
#         self.fid = fid

#         # Vectors
#         self.rivers = rivers
#         self.rivers_main = pd.Series([])

#         # Terrain
#         self.dem = dem.rio.write_nodata(-9999).squeeze()
#         self.dem = self.dem.to_dataset(name='elevation')
#         self.dem.encoding = dem.encoding


# class Reservoir(object):
#     def __init__(self):
#         pass
