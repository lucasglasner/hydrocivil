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
from shapely.geometry import Point
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

from typing import Any, Type, Tuple
from osgeo import gdal
from scipy.interpolate import interp1d

from .misc import polygonize
from .misc import sharegrids, raster_distribution, raster_cross_section
from .unithydrographs import LumpedUnitHydrograph as SUH
from .geomorphology import basin_outlet, process_gdaldem
from .geomorphology import terrain_exposure, get_main_river
from .global_vars import GDAL_EXCEPTIONS
from .abstractions import (cn_correction, SCS_EffectiveRainfall,
                           SCS_EquivalentCurveNumber)

if GDAL_EXCEPTIONS:
    gdal.UseExceptions()
else:
    gdal.DontUseExceptions()
# ---------------------------------------------------------------------------- #


class HydroDEM:
    """
    A class for processing and storing a Digital Elevation Model (DEM) for
    hydrological analysis.
    """

    def __init__(self, dem: xr.DataArray | xr.Dataset, **kwargs):
        """
        Initializes the HydroDEM class with a given Digital Elevation Model.

        Args:
            dem (xr.DataArray | xr.Dataset): A 2D xr.DataArray representing
                the digital elevation model (DEM). If input is an xr.Dataset
                it should contain a single 2D DataArray with the name
                'elevation'.
        """
        self.dem = dem.squeeze().copy()                      # Store DEM
        self.dem = self.dem.rio.write_nodata(np.nan)         # encode no data
        if isinstance(self.dem, xr.DataArray):
            self.dem = self.dem.to_dataset(name='elevation')

        self.mask_raster = ~np.isnan(self.dem['elevation'])  # No data mask
        self.mask_raster.name = 'mask'
        self.rivers = None

    def _get_dem_resolution(self) -> float:
        """
        Compute digital elevation model resolution
        Returns:
            (float, float): raster resolution in the x-y directions
        """
        dx, dy = self.dem.rio.resolution()
        return abs(dx), abs(dy)

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
        self.expdist = self.get_exposure_distribution()
        self.hypsometric_curve = self.get_hypsometric_curve()

        # Height-derived parameters
        params = pd.Series()
        params['hmin'] = self.dem.elevation.min().item()
        params['hmax'] = self.dem.elevation.max().item()
        params['hmean'] = self.dem.elevation.mean().item()
        params['hmed'] = self.dem.elevation.median().item()
        params['deltaH'] = params['hmax']-params['hmin']
        params['deltaHm'] = params['hmean']-params['hmin']
        params['meanslope'] = self.dem.slope.mean().item()
        self.dem_params = params.copy()

    def _process_flow(self,
                      return_streams: bool = False,
                      vector2geopandas: bool = False,
                      carve_dist: float = 0,
                      flow_method: str = 'd8',
                      **kwargs):
        """
        Processes the flow data using the WhiteboxTools package. This method
        preprocesses the digital elevation model (DEM) to generate hydrological
        flow-related rasters. 

        Args:
            carve_dist (float, optional): Maximum distance to carve when
                breaching. Defaults to 0.
            flow_method (str, optional): Flow direction algorithm used for
                computing flow direction and flow accumulation rasters.
                Defaults to 'd8'. Options include: 'd8', 'rho8', 'dinf', 'fd8',
                'Mdinf', 'Quinn1995', 'Qin2007'.
            **kwargs: Additional keyword arguments to be passed to the
                      `wbDEMpreprocess` function.
        Notes:
            - The `wbDEMpreprocess` function is used to preprocess the DEM and
              generate flow-related rasters.
            - The resulting rasters are merged with the existing DEM data.

        """
        from .geomorphology import wbDEMpreprocess
        rasters, rivers = wbDEMpreprocess(self.dem.elevation,
                                          return_streams=return_streams,
                                          raster2xarray=True,
                                          carve_dist=carve_dist,
                                          flow_method=flow_method,
                                          vector2geopandas=vector2geopandas,
                                          **kwargs)
        ivars = ['elevation', 'slope', 'aspect', 'hillshade']
        self.dem = xr.merge([self.dem[ivars]]+rasters)
        self.rivers = rivers
        if isinstance(self.rivers, gpd.GeoDataFrame):
            self.rivers.crs = f'epsg:{self.dem.rio.crs.to_epsg()}'

    def get_exposure_distribution(self, **kwargs) -> pd.Series:
        """
        Based on aspect values calculates the percentage of the raster area that
        faces each of the eight cardinal and intercardinal directions (N, S, E,
        W, NE, SE, SW, NW).

        Args:
            **kwargs:
                direction_ranges: A dictionary mapping direction labels to
                            tuples defining angular ranges in degrees. Defaults
                            to standard 8-direction bins.
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


class HydroLULC:
    """
    A class for processing and analyzing Land Use Land Cover (LULC) for
    hydrological analysis.
    """

    def __init__(self, lulc: xr.Dataset | xr.DataArray, **kwargs):
        """
        Initializes the HydroLULC class with a given land cover dataset.

        Args:
            lulc (xr.DataArray | xr.Dataset):
                A 2D xarray.DataArray or xr.Dataset representing the digital
                surface land cover properties. Might be a single 2D raster or
                multiple rasters in a dataset.
        """
        if isinstance(lulc, xr.DataArray):
            lulc = lulc.to_dataset(name=lulc.name)
        for var in lulc.variables:
            lulc[var] = lulc[var].squeeze().copy()
            # lulc[var] = lulc[var].where(lulc[var] != -9999)

        if 'cn' in lulc.variables:
            lulc['cn1'] = cn_correction(lulc['cn'], amc='I')
            lulc['cn2'] = cn_correction(lulc['cn'], amc='II')
            lulc['cn3'] = cn_correction(lulc['cn'], amc='III')
            if 'amc' in kwargs.keys():
                lulc['cn'] = cn_correction(lulc['cn'], amc=kwargs['amc'])
        self.lulc = lulc

    def _processrastercounts(self, raster: xr.DataArray,
                             output_type: int = 1) -> pd.DataFrame:
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
            if output_type == 1:
                counts = counts.reset_index().rename({raster.name: 'class'},
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

    def _process_lulc(self, **kwargs):
        """
        Process the land use/land cover (LULC) data to compute area
        distributions and other relevant statistics.
        """
        # LULC derived params
        counts, averages = [], []
        for var in self.lulc.data_vars:
            var = self.lulc[var]
            counts.append(self._processrastercounts(var, output_type=1))
            try:
                averages.append(var.mean().item())
            except Exception as e:
                averages.append(np.nan)
                warnings.warn(f'Runtime Exception: {e}')
        self.lulc_counts = pd.concat(counts, keys=self.lulc.data_vars)
        self.lulc_counts = self.lulc_counts.stack().unstack(1).T
        self.lulc_params = pd.Series(averages, index=self.lulc.data_vars)

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
        cn_counts = self._processrastercounts(self.lulc.cn)
        weights, cn_values = cn_counts['counts'].values, cn_counts['cn'].values
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
        return curve


class RiverBasin(HydroDEM, HydroLULC):
    """
    The RiverBasin class represents a hydrological basin and provides methods
    to compute various geomorphological, hydrological, and terrain properties.
    It integrates geographical data, digital elevation models (DEM), river
    networks, and land cover rasters to derive comprehensive watershed
    characteristics.
    """

    def __init__(self, basin: gpd.GeoSeries | gpd.GeoDataFrame,
                 dem: xr.DataArray | xr.Dataset,
                 lulc: xr.DataArray | xr.Dataset,
                 fid: str | int | float = None,
                 rivers: gpd.GeoSeries | gpd.GeoDataFrame = gpd.GeoDataFrame(),
                 match_kwargs={},
                 **kwargs) -> None:
        """
        Initialize the RiverBasin with basin polygon, DEM, LULC, and rivers.

        Args:
            basin (gpd.GeoSeries | gpd.GeoDataFrame): Watershed polygon.
            dem (xr.DataArray | xr.Dataset): Digital elevation model.
            lulc (xr.DataArray | xr.Dataset): Land cover properties as a 2D
                xarray.DataArray or xr.Dataset.
            fid (str | int | float, optional): Feature ID for the basin.
                Defaults to None (generates a random ID).
            rivers (gpd.GeoSeries | gpd.GeoDataFrame, optional): River network
                segments. Defaults to empty GeoDataFrame.
            match_kwargs (dict, optional): Extra arguments for grid matching.
            **kwargs: Additional keyword arguments.
        """
        # Init basin feature ID
        if fid is not None:
            self.fid = fid
        else:
            self.fid = 'Basin_'+f'{np.random.randint(1e6)}'.zfill(6)

        # Init parent constructor
        HydroDEM.__init__(self, dem=dem, **kwargs)
        HydroLULC.__init__(self, lulc=lulc, **kwargs)

        # Match grids
        self._matchgrids(**match_kwargs)

        # Init vector data
        self.basin = basin.copy()                   # Basin polygon
        self.mask_vector = basin.copy()             # Drainage area mask
        self.rivers = deepcopy(rivers)              # Drainage network

        # Init empty attributes
        self.main_river = gpd.GeoDataFrame()
        self.dem_params = pd.DataFrame([], index=[self.fid], dtype=object)
        self.lulc_params = pd.DataFrame([], index=[self.fid], dtype=object)
        self.flow_params = pd.DataFrame([], index=[self.fid], dtype=object)
        self.geoparams = pd.DataFrame([], columns=[self.fid], dtype=object)

    def _matchgrids(self, **kwargs):
        """
        Ensures that the land use/land cover (LULC) raster grid matches the
        digital elevation model (DEM) grid. If the grids do not share the same
        spatial properties, the LULC raster is reprojected to match the DEM
        using rasterio's `reproject_match` method. Additional reprojection
        parameters can be passed via keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the `reproject_match`
            function.
        """
        nlulc = []
        for var in self.lulc.data_vars:
            if sharegrids(self.dem.elevation, self.lulc[var]):
                nlulc.append(self.lulc[var])
            else:
                nvar = self.lulc[var].rio.reproject_match(self.dem.elevation,
                                                          **kwargs)
                nlulc.append(nvar)
        self.lulc = xr.merge(nlulc)

    def copy(self) -> Type['RiverBasin']:
        """
        Create a deep copy of the class itself
        """
        return deepcopy(self)

    def set_parameter(self, index: str | list,
                      data: Any | list) -> Type['RiverBasin']:
        """
        Simple function to add or fix a parameter to the basin parameters table

        Args:
            index (str | list): parameter name/id or what to put in the table
                index.
            data (Any): data for the new parameter
        """
        if np.isscalar(index):
            self.geoparams.loc[index, :] = data
        else:
            for i, v in zip(index, data):
                self.geoparams.loc[i, :] = v

    def _get_basinoutlet(self) -> Tuple[float, float]:
        """
        This function computes the basin outlet point defined as the
        point of minimum elevation along the basin boundary.

        Returns:
            outlet_y, outlet_x (tuple): Tuple with defined outlet y and x
                coordinates.
        """
        outlet_y, outlet_x = basin_outlet(self.basin, self.dem.elevation)
        self.basin['outlet_x'] = outlet_x
        self.basin['outlet_y'] = outlet_y
        return (outlet_y, outlet_x)

    def _process_geography(self) -> pd.DataFrame:
        """
        With the basin polygon this function computes the "geographical" or
        vector properties of the basin (i.e centroid coordinates, area,
        perimeter and outlet to centroid length.)
        """
        c1 = ('outlet_x' in self.basin.columns)
        c2 = ('outlet_y' in self.basin.columns)
        if not (c1 or c2):
            oy, ox = self._get_basinoutlet()
        else:
            oy, ox = self.basin.outlet_y.item(), self.basin.outlet_x.item()

        # General parameters
        self.set_parameter('EPSG', self.basin.crs.to_epsg())
        self.set_parameter('area', self.basin.area.item()/1e6)
        self.set_parameter('outlet_x', ox)
        self.set_parameter('outlet_y', oy)
        self.set_parameter('centroid_x', self.basin.centroid.x.item())
        self.set_parameter('centroid_y', self.basin.centroid.y.item())
        self.set_parameter('perimeter', self.basin.boundary.length.item()/1e3)

        # Outlet to centroid
        outlet = Point(self.basin.outlet_x.item(), self.basin.outlet_y.item())
        out2cen = self.basin.centroid.distance(outlet)
        self.set_parameter('out2centroidlen', out2cen.item()/1e3)

    def _process_terrain(self, **kwargs):
        """
        Processes the Digital Elevation Model (DEM) for slope, aspect and
        multidirectional hillshade. Save everything in the dem dataset.
        Compute general statistics from the DEM and save them in the parameters
        table. 

        Args:
            **kwargs are common arguments for gdaldem slope, aspect and
            hillshade computation.
        """
        super()._process_terrain(**kwargs)
        self.dem_params.name = self.fid
        self.set_parameter(self.dem_params.index.to_list(),
                           self.dem_params.to_list())

    def _process_lulc(self, **kwargs):
        """
        Process the land use/land cover (LULC) data to compute area
        distributions and other relevant statistics. Save average land class
        values in the parameters table. 
        """
        super()._process_lulc(**kwargs)
        self.lulc_params.name = self.fid
        self.set_parameter(self.lulc_params.index.to_list(),
                           self.lulc_params.to_list())

    def _process_flow(self, preprocess_rivers: bool = False,
                      carve_dist: float = 0, flow_method: str = 'rho8',
                      facc_threshold: float = 1e5,
                      **kwargs):
        """
        Compute river network properties
        Args:
            preprocess_rivers (bool, optional): Whether to compute
                river network from given DEM. Requires whitebox_workflows
                package. Defaults to False.
            **kwargs: Additional arguments for the river network preprocessing
                function.
        """
        # Flow derived params
        if preprocess_rivers:
            super()._process_flow(return_streams=True,
                                  vector2geopandas=True,
                                  carve_dist=carve_dist,
                                  flow_method=flow_method,
                                  facc_threshold=facc_threshold,
                                  **kwargs)

        # Main river
        mainriver = get_main_river(self.rivers)
        self.main_river = mainriver

        # Main river stats
        mriverlen = self.main_river.length.sum()/1e3
        if mriverlen.item() != 0:
            mriverlen = mriverlen.item()
            mriverslope = raster_cross_section(self.dem.slope,
                                               mainriver).mean().item()
        else:
            mriverlen = np.nan
            mriverslope = np.nan

        cumlen = self.rivers.length.sum() / 1e3
        area = self.geoparams.loc['area'].item()
        perim = self.geoparams.loc['perimeter'].item()
        eqperim = 2*np.pi*np.sqrt(area/np.pi)

        self.flow_params = pd.Series([])
        self.flow_params['mriverlen'] = mriverlen
        self.flow_params['mriverslope'] = mriverslope
        self.flow_params['drainage_density'] = cumlen/area
        self.flow_params['gravelius_compactness'] = perim/eqperim
        self.flow_params['horton_shape'] = area/mriverlen**2
        self.flow_params.name = self.fid
        self.set_parameter(self.flow_params.index.to_list(),
                           self.flow_params.to_list())

    def compute_params(self, preprocess_rivers=False,
                       geography_kwargs: dict = {},
                       dem_kwargs: dict = {},
                       lulc_kwargs: dict = {},
                       river_network_kwargs: dict = {}) -> pd.DataFrame:
        """
        Compute basin geomorphological properties:

            1) Geographical properties: centroid coordinates, area, etc.
               See self._process_geography.
            2) Land cover properties: average land cover class values and
               percentage of area belonging to each class.
               See HydroLULC._process_lulc.
            3) Terrain properties: DEM-derived properties like minimum,
               maximum, or mean height, etc.
               See HydroDEM._process_terrain.
            4) Flow derived properties: Main river length using graph theory,
               drainage density, and shape factor.
               See src.geomorphology.get_main_river.

        Args:
            dem_kwargs (dict, optional): Additional arguments for the terrain
                preprocessing function. Defaults to {}.
            geography_kwargs (dict, optional): Additional arguments for the
                geography preprocessing routine. Defaults to {}.
            lulc_kwargs (dict, optional): Additional arguments for the land
                cover preprocessing routine. Defaults to {}.
            river_network_kwargs (dict, optional): Additional arguments for the
                main river finding routine. Defaults to {}.
        """
        if self.geoparams.shape != (0, 1):
            self.geoparams = pd.DataFrame([], columns=[self.fid], dtype=object)

        self._process_geography(**geography_kwargs)  # Geographical parameters
        self._process_terrain(**dem_kwargs)  # Update terrain properties
        self._process_lulc(**lulc_kwargs)  # Update land cover properties
        self._process_flow(preprocess_rivers=preprocess_rivers,
                           **river_network_kwargs)  # Flow derived params

    def clip(self,
             poly_mask: gpd.GeoSeries | gpd.GeoDataFrame,
             raster_mask: xr.DataArray,
             **kwargs: Any):
        """
        Clip watershed data to a specified poly_mask boundary and create a new
        RiverBasin object. This method creates a new RiverBasin instance with
        all data (basin boundary, rivers, DEM, etc) clipped to the given
        poly_mask boundary. It also recomputes all geomorphometric parameters for
        the clipped area.

        Args:
            poly_mask (gpd.GeoSeries | gpd.GeoDataFrame): poly_mask defining
                the clip boundary. Must be in the same coordinate reference
                system (CRS) as the watershed data.
            **kwargs (Any): Additional keyword arguments to pass to
                self.compute_params() method.
        Returns:
            self: A new RiverBasin object containing the clipped data and
                updated parameters.
        Notes:
            - The input poly_mask will be dissolved to ensure a single boundary
            - No-data values (-9999) are filtered out from DEM and CN rasters
            - All geomorphometric parameters are recomputed for the clipped
                area
        """
        poly_mask = poly_mask.dissolve()
        self.mask_vector = poly_mask
        self.mask_raster = raster_mask

        self.basin = self.basin.clip(self.mask_vector)  # Basin
        self.dem = self.dem.where(self.mask_raster)  # DEM
        self.lulc = self.lulc.where(self.mask_raster)  # LULC
        self.rivers = self.rivers.clip(poly_mask)  # Rivers
        self.compute_params(**kwargs)

    def update_snowlimit(self, snowlimit: int | float,
                         clean_perc: float = 0.1,
                         polygonize_kwargs: dict = {},
                         **kwargs: Any):
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
            self.geoparams = self.geoparams*0
            self.mask_raster = xr.DataArray(np.full(self.mask_raster.shape,
                                                    False),
                                            dims=self.mask_raster.dims,
                                            coords=self.mask_raster.coords)
            self.mask_vector = gpd.GeoDataFrame()
        elif snowlimit > max_elev:
            warnings.warn(f"snowlimit: {snowlimit} above hmax: {max_elev}")
            self.compute_params(**kwargs)
            self.mask_vector = self.basin
            self.mask_raster = ~self.dem.elevation.isnull()
        else:
            # Create pluvial area mask
            self.mask_raster = self.dem.elevation <= snowlimit
            nshp = polygonize(self.mask_raster, **polygonize_kwargs)

            # Filter out polygons with less than X% of the basin total area
            valid_areas = nshp.area * 100 / self.basin.area.item() > clean_perc
            nshp = nshp[valid_areas]
            self.mask_vector = nshp

            # Clip and save
            self.clip(self.mask_vector, self.mask_raster, **kwargs)

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
        uh = SUH(method, self.geoparams[self.fid])
        uh = uh.compute(**kwargs)
        self.unithydro = uh
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

        if len(self.main_river) > 0:
            self.main_river.plot(ax=ax0, label='Main River', zorder=2,
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
        if len(self.geoparams.index) > 1:
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
