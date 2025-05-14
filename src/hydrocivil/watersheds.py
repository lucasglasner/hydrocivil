"""
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 2024-08-05 11:11:38
 Modified by: Lucas Glasner,
 Modified time: 2024-08-05 11:11:43
 Description: Main watershed class
 Dependencies:
"""


import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

from typing import Any, Type, Tuple
from osgeo import gdal
from scipy.interpolate import interp1d

from .misc import raster_distribution, polygonize
from .unithydrographs import LumpedUnitHydrograph as SUH
from .geomorphology import basin_outlet, process_gdaldem
from .geomorphology import terrain_exposure, get_main_river
from .geomorphology import basin_geographical_params, basin_terrain_params
from .global_vars import GDAL_EXCEPTIONS, _has_whitebox
from .abstractions import cn_correction
from .abstractions import SCS_EffectiveRainfall, SCS_EquivalentCurveNumber


if GDAL_EXCEPTIONS:
    gdal.UseExceptions()
else:
    gdal.DontUseExceptions()
# ---------------------------------------------------------------------------- #


class HydroDEM:
    """
    A class for processing and storing a Digital Elevation Model (DEM) for 
    hydrological analysis.

    This class is designed to handle and process a raster DEM, which may include 
    missing values (NaNs). It creates a mask to identify valid data points and 
    prepares attributes to store derived data such as the hypsometric curve,
    exposure distribution and auxiliary rasters (aspect, slope, etc).
    """

    def __init__(self, dem: xr.DataArray, process_terrain: bool = True,
                 **gdaldem_kwargs):
        """
        Initializes the HydroDEM class with a given Digital Elevation Model.

        Args:
            dem (xr.DataArray): A 2D xarray.DataArray representing the digital 
                                elevation model (DEM).
            process_terrain (bool): If True, preprocess the DEM to compute
                slope, and aspect.
            **gdaldem_kwargs: Additional common arguments for hillshade, slope
            and aspect calculations. (osgeo.gdal.DEMProcessing for details)
        """
        self.dem = dem.squeeze().copy()                      # Store DEM
        self.dem = self.dem.rio.write_nodata(np.nan)         # encode no data
        self.mask_raster = ~np.isnan(self.dem)               # No data mask
        self.mask_raster.name = 'mask'
        self.dem = self.dem.to_dataset(name='elevation')
        self.hypsometric_curve = self.get_hypsometric_curve()
        # Init terrain derived properties
        if process_terrain:
            self._process_terrain(**gdaldem_kwargs)
            self.expdist = self.get_exposure_distribution()

    def copy(self):
        """
        Create a deep copy of the class itself
        """
        return deepcopy(self)

    def _process_terrain(self, **kwargs):
        """
        Processes the Digital Elevation Model (DEM) for slope, aspect and 
        multidirectional hillshade. Save everything in the dem dataset. 

        Args:
            **kwargs are common arguments for gdaldem slope, aspect and
            hillshade computation. 
        """
        slope = process_gdaldem(self.dem.elevation, 'slope',
                                slopeFormat='percent', **kwargs)
        aspect = process_gdaldem(self.dem.elevation, 'aspect',
                                 zeroForFlat=True, **kwargs)
        hs = process_gdaldem(self.dem.elevation, 'hillshade',
                             multiDirectional=True, **kwargs)

        self.dem = xr.merge([self.dem.elevation, slope / 100, aspect, hs])
        self.dem.attrs = {'standard_name': 'terrain model'}

    def _process_flow(self,
                      carve_dist: float = 0,
                      flow_method: str = 'd8',
                      **kwargs):
        """
        Processes the flow data using the WhiteboxTools package if available.
        This method preprocesses the digital elevation model (DEM) to generate
        hydrological flow-related rasters. If the required package is not
        installed, an ImportError is raised.
        Args:
            carve_dist (float, optional): Maximum distance to carve when
                breaching. Defaults to 0.
            flow_method (str, optional): Flow direction algorithm used for
                computing flow direction and flow accumulation rasters. 
                Defaults to 'd8'. Options include: 'd8', 'rho8', 'dinf', 'fd8',
                'Mdinf', 'Quinn1995', 'Qin2007'.
            **kwargs: Additional keyword arguments to be passed to the 
                      `wbDEMpreprocess` function.
            ImportError: If the 'whitebox_workflows' package is not installed.
        Notes:
            - The `wbDEMpreprocess` function is used to preprocess the DEM and 
              generate flow-related rasters.
            - The resulting rasters are merged with the existing DEM data.
            - If the 'whitebox_workflows' package is not available, the method 
              will raise an ImportError with an appropriate message.
        """
        if _has_whitebox:
            from .wb_tools import wbDEMpreprocess
            rasters, _ = wbDEMpreprocess(self.dem.elevation,
                                         return_streams=False,
                                         raster2xarray=True,
                                         carve_dist=carve_dist,
                                         flow_method=flow_method,
                                         **kwargs)
            self.dem = xr.merge([self.dem]+rasters)
        else:
            text = "Flow processing requieres 'whitebox_workflows' package"
            raise ImportError(text)

    def _get_dem_resolution(self) -> float:
        """
        Compute digital elevation model resolution
        Returns:
            (float, float): raster resolution in the x-y directions
        """
        dx, dy = self.dem.rio.resolution()
        return abs(dx), abs(dy)

    def get_exposure_distribution(self, **kwargs) -> pd.Series:
        """
        Based on aspect values calculates the percentage of the raster area that
        faces each of the eight cardinal and intercardinal directions (N, S, E,
        W, NE, SE, SW, NW).

        Args:
            **kwargs:
                direction_ranges: A dictionary mapping direction labels to tuples
                            defining angular ranges in degrees. Defaults to
                            standard 8-direction bins.
                Additional arguments for pandas.Series constructor

        Returns:
            pd.Series: Exposure distribution.
        """
        return terrain_exposure(self.dem.aspect, **kwargs)

    def get_hypsometric_curve(self, bins: str | int | float = 'auto',
                              **kwargs: Any) -> pd.Series:
        """
        Compute the hypsometric curve of the digital elevation model. The
        hypsometric curve represents the distribution of elevation within the
        basin, expressed as the fraction of the total area that lies below a
        given elevation. (Basically is the empirical cumulative distribution
        function)

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
        return curve.cumsum().drop_duplicates()

    def area_below_height(self, height: int | float, **kwargs: Any
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


class RiverBasin(HydroDEM):
    """
    The RiverBasin class represents a hydrological basin and provides methods 
    to compute various geomorphological, hydrological, and terrain properties. 
    It integrates geographical data, digital elevation models (DEM), river 
    networks, and land cover rasters to derive comprehensive watershed 
    characteristics.

    Key Features:
        - Compute geographical parameters such as centroid coordinates, area, 
          and basin outlet.
        - Process DEM to derive hypsometric curves, slope, aspect, and other 
          terrain properties.
        - Analyze river networks to determine main river length and other flow 
          derived properties.
        - Calculate area distributions of raster properties (e.g., land cover 
          classes, soil types).
        - Generate synthetic unit hydrographs using various methods (e.g., SCS, 
          Gray, Linsley) with optional regional parameters for Chile.
        - Clip watershed data to specified polygon boundaries and update 
          geomorphometric parameters.
        - Update the watershed representation to include only the pluvial 
          portion below a specified snow limit elevation.
        - Visualize watershed characteristics including DEM, basin boundary, 
          rivers, hypsometric curve, and terrain aspect distribution.

    Examples:
        + Compute geomorphometric parameters:
            -> wshed = RiverBasin('mybasin', basin, dem, rivers=rivers, cn=cn)
            -> wshed.compute_params()

        + Use curve number corrected by a wet/dry condition:
            -> wshed = RiverBasin('mybasin', basin, dem, rivers, cn, amc='wet')
            -> wshed.compute_params()

        + Change or add a parameter by hand:
            -> wshed.set_parameter('curvenumber', 100)

        + Compute or check hypsometric curve:
            -> curve = wshed.get_hypsometric_curve(bins='auto')
            -> curve = wshed.hypsometric_curve

        + Check fraction of area below 1400 meters:
            -> fArea = wshed.area_below_height(1400)

        + Get relationship of curve number vs precipitation due to basin land 
          cover heterogeneities:
            -> cn_curve = wshed.get_equivalent_curvenumber()

        + Access basin parameters as a pandas DataFrame:
            -> wshed.params

        + Compute SCS unit hydrograph for rain pulses of 1 hour and prf=484:
            -> wshed.SynthUnitHydro(method='SCS', timestep=1, prf=484)

        + Compute flood hydrograph with a series of rainfall:
            -> flood = wshed.UnitHydro.convolve(rainfall)
    """

    def __init__(self, fid: str | int | float,
                 basin: gpd.GeoSeries | gpd.GeoDataFrame,
                 dem: xr.DataArray,
                 rivers: gpd.GeoSeries | gpd.GeoDataFrame = None,
                 cn: xr.DataArray = None,
                 amc: str = 'II') -> None:
        """
        Initializes the basin with various attributes such as basin identifier,
        watershed polygon, digital elevation model (DEM), river network
        segments, curve number raster, and antecedent moisture condition (AMC).

        Args:
            basin (gpd.GeoSeries | gpd.GeoDataFrame): Watershed polygon
            dem (xr.DataArray): Digital elevation model
            rivers (gpd.GeoSeries | gpd.GeoDataFrame, optional):
                River network segments. Defaults to None.
            cn (xr.DataArray, optional): Curve Number raster. Defaults to None,
                which leads to an empty curve number raster.
            amc (str, optional): Antecedent moisture condition.
                Defaults to 'II'. Options: - 'dry' or 'I',
                                           - 'normal' or 'II',
                                           - 'wet' or 'III'.
        """
        # Init basin feature ID
        self.fid = fid

        # Init vector data
        self.basin = basin.copy()                   # Basin polygon
        self.mask_vector = basin.copy()             # Drainage area mask
        self.rivers = deepcopy(rivers)              # Drainage network
        self.rivers_main = gpd.GeoDataFrame()       # Main channel

        # Init HydroDEM constructor
        HydroDEM.__init__(self, dem=dem, process_terrain=True)

        if cn is not None:
            self.cn = cn.rio.write_nodata(-9999).squeeze().copy()
            self.cn = cn_correction(self.cn, amc=amc)
            self.cn_counts = pd.DataFrame([])
            self.amc = amc
        else:
            self.cn = cn

        self.params = pd.DataFrame([], index=[self.fid], dtype=object)
        self.UnitHydro = None

    def __repr__(self) -> str:
        """
        What to show when invoking a RiverBasin object
        Returns:
            str: Some metadata
        """
        if self.UnitHydro is not None:
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
        return deepcopy(self)

    def set_parameter(self, index: str, value: Any) -> Type['RiverBasin']:
        """
        Simple function to add or fix a parameter to the basin parameters table

        Args:
            index (str): parameter name/id or what to put in the table index
            value (Any): value of the new parameter
        """
        self.params.loc[index, :] = value
        return self

    def _get_basinoutlet(self, n: int = 3) -> Tuple[np.ndarray, np.ndarray]:
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
            c1 = 'outlet_x' not in self.basin.columns
            c2 = 'outlet_y' not in self.basin.columns
            if c1 or c2:
                self._get_basinoutlet(n=n)
            else:
                c3 = self.basin['outlet_x'].item() is None
                c4 = self.basin['outlet_y'].item() is None
                if c3 or c4:
                    self._get_basinoutlet(n=n)

            geoparams = basin_geographical_params(self.fid, self.basin,
                                                  **kwargs)
        except Exception as e:
            geoparams = pd.DataFrame([], index=[self.fid])
            warnings.warn('Geographical Parameters Error:'+f'{e}')
        self.params = pd.concat([self.params, geoparams], axis=1)
        return self

    def _processdem(self) -> Type['RiverBasin']:
        """
        Computes DEM-derived properties for the basin and saves them in the
        params dataframe.
        """
        try:
            # DEM derived params
            terrain_params = basin_terrain_params(self.fid, self.dem)
        except Exception as e:
            terrain_params = pd.DataFrame([], index=[self.fid])
            warnings.warn('PostProcess DEM Error:'+f'{e}')
        self.params = pd.concat([self.params, terrain_params], axis=1)

    def _processrivers(self, preprocess_rivers: bool = False,
                       **kwargs,
                       ) -> Type['RiverBasin']:
        """
        Compute river network properties
        Args:
            preprocess_rivers (bool, optional): Whether to compute 
                river network from given DEM. Requires whitebox_workflows
                package. Defaults to False.
            **kwargs: Additional arguments for the river network preprocessing
                function.
        Returns:
            self: updated class
        """
        # Flow derived params
        if self.rivers is None and preprocess_rivers and _has_whitebox:
            from .wb_tools import wbDEMpreprocess
            rasters, rivers = wbDEMpreprocess(self.dem.elevation,
                                              return_streams=True,
                                              raster2xarray=True,
                                              **kwargs)
            self.dem = xr.merge([self.dem]+rasters)
            self.rivers = rivers

        # Main river
        mainriver = get_main_river(self.rivers)
        self.rivers_main = mainriver

        # Main river stats
        mriverlen = self.rivers_main.length.sum()/1e3
        if mriverlen.item() != 0:
            mriverlen = mriverlen.item()
            dx, dy = self._get_dem_resolution()
            geom = mainriver.buffer(max((dx, dy))).geometry
            mriverslope = self.dem.slope.rio.clip(geom).mean().item()
        else:
            mriverlen = np.nan
            mriverslope = np.nan
        self.params['mriverlen'] = mriverlen
        self.params['mriverslope'] = mriverslope
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
            warnings.warn('Raster counting Error:'+f'{e}')
        return counts

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

    def compute_params(self,
                       dem_kwargs: dict = {},
                       geography_kwargs: dict = {},
                       river_network_kwargs: dict = {}) -> Type['RiverBasin']:
        """
        Compute basin geomorphological properties:
            1) Geographical properties: centroid coordinates, area, etc.
                Details in src.geomorphology.basin_geographical_params routine.
            2) Terrain properties: DEM derived properties like minimum, maximum
                or mean height, etc.
                Details in src.geomorphology.basin_terrain_params.
            3) Flow derived properties: Main river length using graph theory, 
                drainage density and shape factor.
                Details in src.geomorphology.get_main_river routine.
        Args:
            dem_kwargs (dict, optional): Additional arguments for the terrain
                preprocessing function. Defaults to {}.
            geography_kwargs (dict, optional): Additional arguments for the
                geography preprocessing routine. Defaults to {}.
            river_network_kwargs (dict, optional): Additional arguments for the
                main river finding routine. Defaults to {}.
        Returns:
            self: updated class
        """
        if self.params.shape != (1, 0):
            self.params = pd.DataFrame([], index=[self.fid], dtype=object)

        # Geographical parameters
        self._processgeography(**geography_kwargs)

        # Update dem properties
        self._processdem(**dem_kwargs)

        # Flow derived params
        self._processrivers(**river_network_kwargs)

        # Curve number process
        if self.cn is not None:
            self.params['curvenumber'] = self.cn.mean().item()

        # Reorder
        self.params = self.params.T.astype(object)

        return self

    def clip(self, polygon: gpd.GeoSeries | gpd.GeoDataFrame,
             **kwargs: Any) -> Type['RiverBasin']:
        """
        Clip watershed data to a specified polygon boundary and create a new
        RiverBasin object. This method creates a new RiverBasin instance with
        all data (basin boundary, rivers, DEM, etc) clipped to the given
        polygon boundary. It also recomputes all geomorphometric parameters for
        the clipped area.

        Args:
            polygon (gpd.GeoSeries | gpd.GeoDataFrame): Polygon defining
                the clip boundary. Must be in the same coordinate reference
                system (CRS) as the watershed data.
            **kwargs (Any): Additional keyword arguments to pass to
                self.compute_params() method.
        Returns:
            self: A new RiverBasin object containing the clipped data and
                updated parameters.
        Notes:
            - The input polygon will be dissolved to ensure a single boundary
            - No-data values (-9999) are filtered out from DEM and CN rasters
            - All geomorphometric parameters are recomputed for the clipped
              area
        """
        nwshed = self.copy()
        polygon = polygon.dissolve()

        # Basin
        nbasin = self.basin.copy().clip(polygon)
        nwshed.basin = nbasin
        nwshed.mask_vector = nbasin

        # DEM & mask
        ndem = self.dem.copy().rio.clip(polygon.geometry)
        ndem = ndem.where(ndem != -9999)
        ndem = ndem.reindex({'y': self.dem.y, 'x': self.dem.x})
        nmask = ~np.isnan(ndem.elevation)
        nmask.name = 'mask'
        nwshed.dem = ndem
        nwshed.mask_raster = nmask

        # Rivers
        if self.rivers is not None:
            nrivers = self.rivers.copy().clip(polygon)
            nwshed.rivers = nrivers

        # Curve Number
        if self.cn is not None:
            ncn = self.cn.copy().rio.clip(polygon.geometry)
            ncn = ncn.where(ncn != -9999)
            ncn = ncn.reindex({'y': self.cn.y, 'x': self.cn.x})
            nwshed.cn = ncn

        nwshed.compute_params(**kwargs)
        return nwshed

    def update_snowlimit(self, snowlimit: int | float,
                         clean_perc: float = 0.1,
                         polygonize_kwargs: dict = {},
                         **kwargs: Any) -> Type['RiverBasin']:
        """
        Updates the RiverBasin object to represent only the pluvial (rain-fed) 
        portion of the watershed below a specified snow limit elevation.

        This method clips the basin to areas below the given snow limit 
        elevation threshold. The resulting watershed represents only the 
        portion of the basin that receives precipitation as rainfall rather 
        than snow. All watershed properties (e.g., area, rivers, DEM, etc.) 
        are updated accordingly.

        Args:
            snowlimit (int|float): Elevation threshold (in the same units as 
            the DEM) that defines the rain/snow transition zone.
            clean_perc (float): Minimum polygon area (as a percentage of the 
                total basin area) to be included in the pluvial zone. Defaults
                to 0.1%. 
            polygonize_kwargs (dict, optional): Additional keyword arguments 
            passed to the polygonize function. Defaults to {}.
            **kwargs: Additional keyword arguments passed to the compute_params 
            method.

        Raises:
            TypeError: If the snowlimit argument is not numeric.

        Returns:
            RiverBasin: The updated RiverBasin object containing only the 
            pluvial portion of the original watershed below the snow limit.
        """
        if not isinstance(snowlimit, (int, float)):
            raise TypeError("snowlimit must be numeric")
        min_elev = self.dem.elevation.min().item()
        max_elev = self.dem.elevation.max().item()
        if snowlimit < min_elev:
            warnings.warn(f"snowlimit: {snowlimit} below hmin: {min_elev}")
            self.params = self.params*0
            self.mask_raster = xr.DataArray(np.full(self.mask_raster.shape,
                                                    False),
                                            dims=self.mask_raster.dims,
                                            coords=self.mask_raster.coords)
            self.mask_vector = gpd.GeoDataFrame()
            return self
        elif snowlimit > max_elev:
            warnings.warn(f"snowlimit: {snowlimit} above hmax: {max_elev}")
            self.compute_params(**kwargs)
            self.mask_vector = self.basin
            self.mask_raster = ~self.dem.elevation.isnull()
            return self
        else:
            # Create pluvial area mask
            mask = self.dem.elevation <= snowlimit
            nshp = polygonize(mask, **polygonize_kwargs)

            # Filter out polygons with less than X% of the basin total area
            valid_areas = nshp.area * 100 / self.basin.area.item() > clean_perc
            nshp = nshp[valid_areas]

            # Clip and save
            nwshed = self.clip(nshp, **kwargs)
            self.params = nwshed.params
            self.mask_raster = nwshed.mask_raster
            self.mask_vector = nwshed.mask_vector
            return self

    def SynthUnitHydro(self, method: str, **kwargs: Any) -> Type['RiverBasin']:
        """
        Compute synthetic unit hydrograph for the basin.

        This method creates and computes a synthetic unit hydrograph based
        on basin parameters. For Chilean watersheds, special regional
        parameters can be used if ChileParams = True.

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
            RiverBasin: Updated instance with computed unit hydrograph stored
                in UnitHydro attribute.

        Raises:
            RuntimeError: If using Chilean parameters and basin centroid lies
                outside valid geographical regions.
        """
        uh = SUH(method, self.params[self.fid])
        uh = uh.compute(**kwargs)
        self.UnitHydro = uh
        return self

    def plot(self,
             demvar='elevation',
             legend_kwargs: dict = {'loc': 'upper left'},
             outlet_kwargs: dict = {'ec': 'k', 'color': 'tab:red'},
             basin_kwargs: dict = {'edgecolor': 'k'},
             demimg_kwargs: dict = {'cbar_kwargs': {'shrink': 0.8}},
             mask_kwargs: dict = {'hatches': ['////']},
             demhist_kwargs: dict = {'alpha': 0.5},
             hypsometric_kwargs: dict = {'color': 'darkblue'},
             rivers_kwargs: dict = {'color': 'tab:red'},
             exposure_kwargs: dict = {'ec': 'k', 'width': 0.6},
             kwargs: dict = {'figsize': (12, 5)}) -> matplotlib.axes.Axes:
        """
        Create a comprehensive visualization of watershed characteristics
            including:
            - 2D map view showing DEM, basin boundary, rivers and outlet point
            - Polar plot showing terrain aspect/exposure distribution
            - Hypsometric curve and elevation histogram

        Args:
            legend (bool, optional): Whether to display legend.
                Defaults to True.
            legend_kwargs (dict, optional): Arguments for legend formatting.
                Defaults to {'loc': 'upper left'}.
            outlet_kwargs (dict, optional): Styling for basin outlet point.
                Defaults to {'ec': 'k', 'color': 'tab:red'}.
            basin_kwargs (dict, optional): Styling for basin boundary.
                Defaults to {'edgecolor': 'k'}.
            demimg_kwargs (dict, optional): Arguments for DEM image display.
                Defaults to {'cbar_kwargs': {'shrink': 0.8}}.
            mask_kwargs  (dict, optional): Arguments for mask hatches.
                Defaults to {'hatches': ['////']}.
            demhist_kwargs (dict, optional): Arguments for elevation histogram.
                Defaults to {'alpha': 0.5}.
            hypsometric_kwargs (dict, optional): Styling for hypsometric curve.
                Defaults to {'color': 'darkblue'}.
            rivers_kwargs (dict, optional): Styling for river network.
                Defaults to {'color': 'tab:red'}.
            exposure_kwargs (dict, optional): Styling for polar exposure plot
                Defaults to {'ec':'k', 'width':0.5}
            kwargs (dict, optional): Additional figure parameters.
                Defaults to {'figsize': (12, 5)}.

        Returns:
            (tuple): Matplotlib figure and axes objects
                (fig, (ax0, ax1, ax2, ax3))
                - ax0: Map view axis
                - ax1: Aspect distribution polar axis  
                - ax2: Hypsometric curve axis
                - ax3: Elevation histogram axis
        """
        # Create figure and axes
        fig = plt.figure(**kwargs)
        ax0 = fig.add_subplot(121)
        ax1 = fig.add_subplot(222, projection='polar')
        ax2 = fig.add_subplot(224)
        ax3 = ax2.twinx()

        # Plot basin and rivers
        try:
            self.basin.boundary.plot(ax=ax0, zorder=2, **basin_kwargs)
            ax0.scatter(self.basin['outlet_x'], self.basin['outlet_y'],
                        label='Outlet', zorder=3, **outlet_kwargs)
        except Exception as e:
            warnings.warn(str(e))

        if len(self.rivers_main) > 0:
            self.rivers_main.plot(ax=ax0, label='Main River', zorder=2,
                                  **rivers_kwargs)

        # Plot dem data
        try:
            self.dem[demvar].plot.imshow(ax=ax0, zorder=0, **demimg_kwargs)
            if len(self.hypsometric_curve) == 0:
                self.get_hypsometric_curve()
            hypso = self.hypsometric_curve
            hypso.plot(ax=ax2, zorder=1, label='Hypsometry',
                       **hypsometric_kwargs)
            ax3.plot(hypso.index, hypso.diff(), zorder=0, **demhist_kwargs)
        except Exception as e:
            warnings.warn(str(e))

        # Plot snow area mask
        try:
            mask = self.mask_raster
            nanmask = self.dem.elevation.isnull()
            if (~nanmask).sum().item() != mask.sum().item():
                mask.where(~nanmask).where(~mask).plot.contourf(
                    ax=ax0, zorder=1, colors=None, alpha=0, add_colorbar=False,
                    **mask_kwargs)
                ax0.plot([], [], label='Snowy Area', color='k')
        except Exception as e:
            warnings.warn(str(e))

        # Plot basin exposition
        if len(self.params.index) > 1:
            exp = pd.DataFrame(self.expdist, columns=[self.fid])
            exp.index = exp.index.map(lambda x: x.split('_')[0])
            exp = exp.loc[['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']]
            exp = pd.concat([exp.iloc[:, 0], exp.iloc[:, 0][:'N']])
            ax1.bar(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315, 0]), exp,
                    **exposure_kwargs)
            ax1.set_xticks(ax1.get_xticks())
            ax1.set_xticklabels(exp.index.values[:-1])
            ax1.set_ylim(0, exp.max()*1.1)

        # Aesthetics
        try:
            for axis in [ax0, ax1, ax2, ax3]:
                axis.set_title('')
                if axis in [ax0, ax2]:
                    axis.legend(**legend_kwargs)
            bounds = self.basin.minimum_bounding_circle().bounds
            ax0.set_xlim(bounds.minx.item(), bounds.maxx.item())
            ax0.set_ylim(bounds.miny.item(), bounds.maxy.item())
            ax1.set_theta_zero_location("N")
            ax1.set_theta_direction(-1)
            ax1.set_xticks(ax1.get_xticks())
            ax1.set_yticklabels([])
            ax1.grid(True, ls=":")

            ax2.grid(True, ls=":")
            ax2.set_ylim(0, 1)
            ax3.set_ylim(0, ax3.get_ylim()[-1])
            ax2.set_xlabel('(m)')

        except Exception as e:
            warnings.warn(str(e))
        return fig, (ax0, ax1, ax2, ax3)
