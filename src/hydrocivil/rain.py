'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 1969-12-31 21:00:00
 Modified by: Lucas Glasner,
 Modified time: 2024-05-06 16:24:28
 Description:
 Dependencies:
'''

import numpy as np
import pandas as pd
import copy as pycopy
import xarray as xr

from typing import Any, Type
from numpy.typing import ArrayLike
from .global_vars import SHYETO_DATA, DURCOEFS_PGAUGES
from .misc import obj2xarray, series2func, rsquared, alternating_block_sort
import warnings

import scipy.stats as st
from scipy.optimize import curve_fit


# -------------------- Intensity-duration-frequency curves ------------------- #
def grunsky_coef(storm_duration: float | ArrayLike,
                 expon: float = 0.5) -> float:
    """
    This function computes the duration coefficient given by a Grunsky-like
    Formula. Those formulas state that the duration coefficient is a power
    law of the storm duration t:

        Cd (t) = (t / 24) ^ b

    Where "t" is the storm duration of interest in hours, and "b" is an
    empirical parameter. The traditional Grunsky formula assumes b = 0.5,
    which is generally valid for cyclonic precipitation on flat terrain24.
    However, for convective rainfall or rainfall on complex terrain24, a
    different power-law may apply (or may not).

    References:
        ???
        Stowhas, Ludwing (2017). Fundamentos de Hidrología Aplicada. Editorial
        Universidad Federico Santa María. Valparaíso, Chile.

    Args:
        storm_duration (float | ArrayLike): storm duration in (hours)
        expon (float): Exponent of the power law. Defaults to 0.5 (Grunsky).

    Returns:
        CD (float | ArrayLike): Duration coefficient (dimensionless)
    """
    return (storm_duration/24)**expon


def bell_coef(storm_duration: float | ArrayLike,
              cd1: float | ArrayLike) -> float | ArrayLike:
    """
    This function computes the duration coefficient given by the Bell Formula.
    These formula state that the duration coefficient is given by:

        Cd (t) = (0.54 * (t * 60)^0.25 - 0.5) * Cd(1)

    Where "t" is the storm duration of interest in hours, and Cd(1) is the
    duration coefficient for a 1 hour storm duration. This formula is generally
    valid for very short storms of duration less than 1 hour.

    References:
        Bell, F.C. (1969) Generalized pr-Duration-Frequency
        Relationships. Journal of Hydraulic Division, ASCE, 95, 311-327.

    Args:
        storm_duration (float | ArrayLike): duration in (hours)
        cd1 (float | ArrayLike): Duration coefficient for 1 hour duration.
    Returns:
        CD (float | ArrayLike): Duration coefficient (dimensionless)
    """
    storm_duration = np.asarray(storm_duration, dtype=float)
    if np.any(storm_duration >= 1):
        raise ValueError("Bell formula is only valid for durations < 1 hr.")
    a = (0.54*((storm_duration*60)**0.25)-0.5)
    a = np.where(a < 0, 0, a)
    return a * cd1


def validate_duration(storm_duration: float | ArrayLike) -> None:
    """
    Validate storm duration input.
    """
    durations = np.asarray(storm_duration, dtype=float)
    if np.any(durations < 0):
        raise ValueError("Storm duration must be positive.")


def duration_coef(storm_duration: int | float, ref_pgauge: str = 'Grunsky',
                  expon: float = 0.5, bell_threshold: int | float = 1,
                  **kwargs) -> float:
    """
    Compute duration coefficient to convert 24 hour precipitation to a
    target duration. Used to estimate rainfall for different durations based on
    empirical formulas. By default:

    - For durations >= 1 hr, Grunsky's formula is used.
    - For durations < 1 hr, Bell's formula is used.
    - If a reference rain gauge is specified (other than 'Grunsky'),
      coefficients are interpolated from predefined tables. Bell's formula is
      still applied for durations < 1 hr and can be controled with the
      bell_threshold parameter.

    Args:
        storm_duration (int, float, or array-like): Target duration(s) [hr].
        ref_pgauge (str, optional): Reference gauge name or 'Grunsky' method.
            Options: Putre, Lequena, Toconce, Rivadavia, La Paloma, Illapel,
            La Tranquilla, Quillota, Rungue, Lago Peñuelas, Los Panguiles,
            Quinta Normal, San Joaquin, Pirque, Melipilla, Rapel, Llaullauquén,
            San Fernando, Curicó, Armerillo, Colbún, Chillán, Concepción,
            Polcura, Quilaco, Temuco, Pullinque, Valdivia, Osorno, Ensenada,
            Puerto Montt, Lago Chapo, Canutillar, Chaitén, Puerto Aysén,
            Punta Arenas. Default is 'Grunsky'.
        expon (float): Exponent of the Grunsky power law. Defaults to 0.5.
        bell_threshold (int, float): Duration threshold [hr] below which
            Bell's formula is applied. Default is 1 hr. Force to 0 to disable.
    Returns:
        np.ndarray: Duration coefficient(s) (dimensionless).
    """
    validate_duration(storm_duration)
    validate_duration(bell_threshold)

    durations = np.asarray(storm_duration, dtype=float)
    scalar_input = durations.ndim == 0
    durations = np.atleast_1d(durations)

    bell_mask = durations < bell_threshold
    if ref_pgauge == 'Grunsky':
        coefs = grunsky_coef(durations, expon=expon)
        cd1 = grunsky_coef(1.0, expon=expon)
    else:
        if ref_pgauge not in DURCOEFS_PGAUGES.columns:
            raise ValueError(f"Reference rain gauge '{ref_pgauge}' not found.")
        if durations.max() > DURCOEFS_PGAUGES.index.max():
            warnings.warn(
                f"Maximum duration {durations.max()} hr exceeds "
                f"table maximum of {DURCOEFS_PGAUGES.index.max()} hr for gauge "
                f"'{ref_pgauge}'. Extrapolating beyond table bounds.",
                UserWarning
            )
        func = series2func(DURCOEFS_PGAUGES[ref_pgauge], **kwargs)
        coefs = func(durations)
        cd1 = DURCOEFS_PGAUGES[ref_pgauge].loc[1]

    if bell_mask.any():
        coefs = coefs.copy()
        coefs[bell_mask] = bell_coef(durations[bell_mask], cd1=cd1)

    coefs = np.clip(coefs, 0, None)
    return coefs if not scalar_input else coefs[0]


class IDF:
    """
    General Intensity-Duration-Frequency (IDF) relationship handler.
    """

    def __init__(self,
                 timestep: float,
                 rain24: ArrayLike = None,
                 idf: ArrayLike = None,
                 duration_limits: ArrayLike = (0, 72)) -> None:
        """
        Intensity-Duration-Frequency (IDF) relationship handler.

        Args:
            timestep (float): Time step or resolution in hours.
            rain24 (ArrayLike, optional): Rainfall amount(s) in mm per 24 hours.
            idf (ArrayLike, optional): IDF curve as an array with an
                identifiable 'duration' dimension. If None, IDF will be computed
                from rainfall using duration_coef. Defaults to None.
            duration_limits (ArrayLike, optional): Min and max duration limits
                in hours for the IDF curve. Defaults to (0, 72).
        """
        # Validate inputs
        self._validate_inputs(timestep, rain24, idf, duration_limits)

        # Convert rain24 and idf to xarray DataArrays
        xr_rain24 = obj2xarray(
            rain24, name='rain24') if rain24 is not None else None
        xr_idf = obj2xarray(
            idf, name='idf') if idf is not None else None

        if rain24 is None and idf is not None:
            self.idf_computation_path = 'idf-from-sample'
            xr_idf = self._validate_idf_curve(xr_idf)
        else:
            self.idf_computation_path = 'idf-from-rain24'

        # Store input arguments
        self.timestep = timestep
        self.rain24 = xr_rain24
        self.idf = None
        self.rdf = None
        self.dcoefs = None
        self.duration_limits = duration_limits

        # Build aditional attributes
        self.durations = self._build_duration_axis()
        self.idf_sampled = xr_idf
        self.rdf_sampled = None
        self.dcoefs_sampled = None
        self.rsquared_ = None
        self.fitcoefs_ = None
        self.parametric_ = None
        self.idf_model_type_ = None
        self.bell_threshold_ = None

    # -------------------------------- Validations --------------------------- #
    def _validate_inputs(self,
                         timestep: float,
                         rain24: ArrayLike,
                         idf: ArrayLike,
                         duration_limits: ArrayLike):
        """
        Validate input parameters for IDF class.
        """
        if (rain24 is not None) == (idf is not None):
            raise ValueError("Must specify exactly one of 'rain24' or 'idf'")

        validate_duration(duration_limits)

        if timestep <= 0:
            raise ValueError(f"timestep must be positive, got {timestep}")

        if np.array(duration_limits).ndim != 1 or len(duration_limits) != 2:
            raise ValueError("duration_limits must be a 1D array of length 2")

        if duration_limits[1] < 24:
            raise ValueError("Maximum duration limit should be at least 24 hr")

    def _validate_idf_curve(self, xr_idf_curve: xr.DataArray) -> xr.DataArray:
        """
        Validate IDF curve structure and sort by duration if needed.
        """
        if 'duration' not in xr_idf_curve.dims:
            raise ValueError("idf_curve must have a 'duration' dimension")
        if (xr_idf_curve < 0).any():
            raise ValueError("IDF curve values must be non-negative")
        # Ensure monotonic increasing duration coordinate
        if not xr_idf_curve['duration'].to_series().is_monotonic_increasing:
            xr_idf_curve = xr_idf_curve.sortby('duration')
        return xr_idf_curve

    # -------------------------------- Conversions --------------------------- #
    def _idf2rdf(self, xr_idf: xr.DataArray) -> xr.DataArray:
        """Convert IDF to RDF curve."""
        xr_rdf = xr_idf * xr_idf['duration']
        xr_rdf.name = 'rainfall'
        xr_rdf.attrs = {'standard_name': 'total_rainfall_vs_duration',
                        'units': 'mm'}
        return xr_rdf

    def _rdf2idf(self, xr_rdf: xr.DataArray) -> xr.DataArray:
        """Convert RDF to IDF curve."""
        durations = xr_rdf['duration']
        xr_idf = xr_rdf / durations
        xr_idf.name = 'intensity'
        xr_idf.attrs = {'standard_name': 'intensity_vs_duration',
                        'units': 'mm/hr'}
        return xr_idf

    def _idf2dcoef(self, xr_idf: xr.DataArray) -> xr.DataArray:
        """Convert IDF to duration coefficient."""
        durations = xr_idf['duration']
        cds = xr_idf * durations
        cds = cds / cds.interp(duration=24)
        cds.name = 'dcoef'
        cds.attrs = {'standard_name': 'duration_coefficient',
                     'units': '-'}
        return cds

    def _rdf2dcoef(self, xr_rdf: xr.DataArray) -> xr.DataArray:
        """Convert RDF to duration coefficient."""
        cds = xr_rdf / xr_rdf.interp(duration=24)
        cds.name = 'dcoef'
        cds.attrs = {'standard_name': 'duration_coefficient',
                     'units': '-'}
        return cds

    def _force0duration(self) -> xr.DataArray:
        """
        Force duration coefficient at 0 duration to be 0.
        Force RDF at 0 duration to be 0.
        Force IDF at 0 duration to be np.nan.
        """
        self.idf.loc[{'duration': 0}] = np.nan
        self.rdf.loc[{'duration': 0}] = 0
        self.dcoefs.loc[{'duration': 0}] = 0

    # ----------------------------- Parametric Models ------------------------ #
    @staticmethod
    def _powerlaw_type1(t, b, rain_24):
        """
        Type I power law model with constraint i(24) = rain_24/24
        i(t) = a / t^b, where a = rain_24 * 24^(b-1)
        """
        a = rain_24 * (24 ** (b - 1))
        return a / (t ** b)

    @staticmethod
    def _powerlaw_type2(t, b, c, rain_24):
        """
        Type II power law model with constraint i(24) = rain_24/24
        i(t) = a / (t^b + c), where a = rain_24 * (24^b + c) / 24
        """
        a = rain_24 * (24 ** b + c) / 24
        return a / (t ** b + c)

    @staticmethod
    def _powerlaw_type3(t, b, c, rain_24):
        """
        Type III power law model with constraint i(24) = rain_24/24
        i(t) = a / (t + c)^b, where a = rain_24 * (24 + c)^b / 24
        """
        a = rain_24 * ((24 + c) ** b) / 24
        return a / ((t + c) ** b)

    # ----------------------------------- Core ------------------------------- #
    def _build_duration_axis(self):
        """
        Build time axis based on duration limits and timestep.
        """
        dt = self.timestep
        dur_min, dur_max = self.duration_limits
        durations = np.arange(dur_min, dur_max + dt, dt)
        durations = xr.DataArray(durations, dims=['duration'])
        return durations

    def _sample_idf_curve(self, durations: ArrayLike = None,
                          **kwargs) -> xr.DataArray:
        """
        Build the IDF and RDF curve based on rain and duration coeficients
        """
        # Build
        if durations is None:
            durations = DURCOEFS_PGAUGES.index.values
        durations = xr.DataArray(durations, dims=['duration'])
        cds = duration_coef(durations, **kwargs)
        cds = xr.DataArray(cds, dims=['duration'], name='cds',
                           coords={'duration': durations})
        rdf = self.rain24.expand_dims({'duration': durations}, axis=0) * cds
        idf = rdf / rdf['duration']

        # Name and attrs
        rdf.name = 'rainfall'
        rdf.attrs = {'standard_name': 'total_rainfall_vs_duration',
                     'units': 'mm'}

        idf.name = 'intensity'
        idf.attrs = {'standard_name': 'intensity_vs_duration',
                     'units': 'mm/hr'}

        self.idf_sampled = idf
        self.rdf_sampled = rdf
        self.dcoefs_sampled = self._rdf2dcoef(rdf)
        return idf, rdf

    def _fit_parametric_model(self, idf_model_type: int, **kwargs):
        """
        Fit a parametric IDF model to the sampled IDF curve and store the
        fit coefficients and R-squared value.

        The fit enforces the constraint i(24) = rain24/24, reducing the number
        of free parameters by one.
        """
        func_map = {
            1: (self._powerlaw_type1, [[], []]),      # Only b parameter + r2
            2: (self._powerlaw_type2, [[], [], []]),  # b, c parameters + r2
            3: (self._powerlaw_type3, [[], [], []])   # b, c parameters + r2
        }

        func, output_dims = func_map[idf_model_type]

        def _func2fit(i, t, rain_24, **kwargs):
            # Create wrapper with explicit parameters based on model type
            if idf_model_type == 1:
                # Type I: only b parameter
                def func_with_rain_24(
                    t_fit, b): return func(t_fit, b, rain_24)
            else:
                # Type II and III: b, c parameters
                def func_with_rain_24(t_fit, b, c): return func(
                    t_fit, b, c, rain_24)

            popt, _ = curve_fit(func_with_rain_24, t, i,
                                bounds=(0, np.inf), **kwargs)
            return (*popt, rsquared(i, func(t, *popt, rain_24)))

        # Get rain24 at 24 hours
        rain_24 = self.idf_sampled.interp(duration=24) * 24

        output = xr.apply_ufunc(_func2fit,
                                self.idf_sampled,
                                self.idf_sampled['duration'],
                                rain_24,
                                input_core_dims=[
                                    ['duration'], ['duration'], []],
                                output_core_dims=output_dims,
                                vectorize=True,
                                kwargs=kwargs)
        popt = output[:-1]
        r2 = output[-1]
        self.rsquared_ = r2
        self.fitcoefs_ = popt
        return popt, r2

    def _force_bell_threshold(self, idf: xr.DataArray,
                              bell_threshold: float) -> xr.DataArray:
        """
        Force Bell threshold application on the IDF curve.
        """
        if bell_threshold == 0:
            return idf
        name, attrs = idf.name, idf.attrs
        rdf = self._idf2rdf(idf)
        dcoef = self._rdf2dcoef(rdf)

        bell_mask = idf['duration'] < bell_threshold
        if bell_mask.any():
            t, cd1 = xr.broadcast(idf['duration'][bell_mask],
                                  dcoef.interp(duration=1))
            dcoef_bell = bell_coef(t, cd1)
            dcoef.loc[{'duration': idf['duration'][bell_mask]}] = dcoef_bell
            idf = dcoef * rdf.interp(duration=24) / idf['duration']
            idf.name = name
            idf.attrs = attrs
        return idf

    def _evaluate_parametric_model(self, durations: ArrayLike,
                                   idf_model_type: int) -> xr.DataArray:
        """
        Evaluate the fitted parametric model at given durations.
        """
        durations = xr.DataArray(durations, dims=['duration'])
        if self.fitcoefs_ is None and self.rsquared_ is None:
            raise ValueError("Model must be fitted before evaluation.")

        model_map = {
            1: self._powerlaw_type1,
            2: self._powerlaw_type2,
            3: self._powerlaw_type3
        }

        model = model_map[idf_model_type]

        # Get rain24 for a 24 hour storm
        rain_24 = self.idf_sampled.interp(duration=24) * 24

        # Evaluate model with rain_24 constraint
        idf = model(durations, *self.fitcoefs_, rain_24)
        idf = idf.assign_coords({'duration': durations})
        idf = idf.transpose(*self.idf_sampled.dims)
        idf.name = 'intensity'
        idf.attrs = {'standard_name': 'intensity_vs_duration',
                     'units': 'mm/hr'}
        return idf

    def fit(self, parametric=True, idf_model_type: int = 2,
            bell_threshold: float = 1, **kwargs) -> Type['IDF']:
        """
        Fit the Intensity-Duration curve using the specified computation path.
        This method fits an IDF curve based on either rainfall data or existing
        IDF data, applies a parametric model, and computes related rainfall
        properties including RDF(Rainfall-Duration) and disaggregation/duration
        coefficients.

        Args:
            parametric (bool): If True, fits a parametric model to the IDF
                curve. If False, interpolate/extrapolate the sampled IDF curve
                directly. Defaults to True.
            idf_model_type (int, optional): Type of parametric model to fit.
                Options:
                    1: Type I power law: i(t) = a / t^b
                    2: Type II power law: i(t) = a / (t^b + c)
                    3: Type III power law: i(t) = a / (t + c)^b
                Defaults to 2.
            bell_threshold (int, float): Duration threshold [hr] below which
                Bell's formula is applied. Default is 1 hr. Force to 0 to
                disable.
            **kwargs: Additional keyword arguments for duration_coef if rain24
                is provided (e.g., ref_pgauge, expon, etc.).

        Returns:
            IDF: The instance itself with updated attributes.
        Raises:
            AttributeError: If required attributes for the specified
                idf_computation_path are not properly initialized in the class
                constructor.
        """
        # Validate idf_model_type
        if idf_model_type not in [1, 2, 3]:
            raise ValueError("idf_model_type must be 1, 2, or 3")

        # Store input parameters for futher reference
        self.parametric_ = parametric
        self.idf_model_type_ = idf_model_type
        self.bell_threshold_ = bell_threshold

        if self.idf_computation_path == 'idf-from-rain24':
            if 'ref_pgauge' in kwargs and kwargs['ref_pgauge'] != 'Grunsky':
                if parametric:
                    self._sample_idf_curve(**kwargs)
                    self._fit_parametric_model(idf_model_type)
                    idf = self._evaluate_parametric_model(self.durations,
                                                          idf_model_type)
                else:
                    self._sample_idf_curve(self.durations, **kwargs)
                    idf = self.idf_sampled
                idf = self._force_bell_threshold(idf, bell_threshold)
                self.idf = idf
                self.rdf = self._idf2rdf(self.idf)
                self.dcoefs = self._idf2dcoef(self.idf)
            else:
                self._sample_idf_curve(self.durations,
                                       bell_threshold=bell_threshold, **kwargs)
                self.idf = self.idf_sampled
                self.rdf = self.rdf_sampled
                self.dcoefs = self.dcoefs_sampled

        if self.idf_computation_path == 'idf-from-sample':
            self.rdf_sampled = self._idf2rdf(self.idf_sampled)
            self.dcoefs_sampled = self._idf2dcoef(self.idf_sampled)
            if parametric:
                self._fit_parametric_model(idf_model_type)
                idf = self._evaluate_parametric_model(self.durations,
                                                      idf_model_type)
            else:
                # Check if extrapolation is needed
                sampled_max = float(self.idf_sampled['duration'].max())
                sampled_min = float(self.idf_sampled['duration'].min())
                cond = (self.durations > sampled_max).any()
                cond = cond or (self.durations < sampled_min).any()
                if cond:
                    warnings.warn(
                        f"Extrapolating beyond sampled duration bounds "
                        f"[{sampled_min}, {sampled_max}] hr. Results may be "
                        f"unreliable. Consider fitting a parametric model.",
                        UserWarning
                    )
                idf = self.idf_sampled.interp(
                    duration=self.durations,
                    kwargs={"fill_value": "extrapolate"})
            idf = self._force_bell_threshold(idf, bell_threshold)
            self.idf = idf
            self.rdf = self._idf2rdf(self.idf)
            self.dcoefs = self._idf2dcoef(self.idf)

        self._force0duration()
        return self


# ------------------------------- Design Storms ------------------------------ #
class RainStorm(IDF):
    """
    Generic class to build rainstorms that follow any of scipy theoretical
    distributions (e.g 'norm', 'skewnorm', 'gamma', etc), empirical synthetic
    hyetographs (e.g SCS Type I, G2_Espildora1979, G4p50_Varas1985, etc), or
    specific design storm methods like triangular, alternating block or instant
    intensity hyetographs (Chow, 1995).
    """
    PREDEFINED_STORMS = SHYETO_DATA.columns

    def __init__(self, storm_type: str, timestep: float,
                 rain24: float | ArrayLike = None, idf: ArrayLike = None,
                 idf_scaling: bool = True, duration_limits: ArrayLike = (0, 72),
                 **storm_kwargs) -> None:
        """
        Synthetic RainStorm builder

        Args:
            storm_type (str): Type of storm model to use.
                - Options:
                   G1_Espildora1979, G2_Espildora1979, G3_Espildora1979,
                   G1_Benitez1985, G2_Benitez1985, G3_Benitez1985,
                   SCS_I24, SCS_IA24,
                   SCS_II6, SCS_II12, SCS_II24, SCS_II48,
                   SCS_III24,
                   G1p10_Varas1985, G1p25_Varas1985, G1p50_Varas1985,
                   G1p75_Varas1985, G1p90_Varas1985,
                   G2p10_Varas1985, G2p25_Varas1985, G2p50_Varas1985,
                   G2p75_Varas1985, G2p90_Varas1985,
                   G3p10_Varas1985, G3p25_Varas1985, G3p50_Varas1985,
                   G3p75_Varas1985, G3p90_Varas1985,
                   G4p10_Varas1985, G4p25_Varas1985, G4p50_Varas1985,
                   G4p75_Varas1985, G4p90_Varas1985,
                - SciPy distribution (e.g., 'norm', 'gamma')
                - alternating_block
            timestep (float): Time step or resolution in hours.
            rain24 (ArrayLike, optional): rainfall amount(s) in mm per 24 hours.
            idf (ArrayLike, optional): IDF curve as an array with an
                identifiable 'duration' dimension. If None, IDF will be computed
                from rainfall using duration_coef. Defaults to None.
            idf_scaling (bool): If True, scale the hyetograph to match the
                intensity-duration curve at the specified rain24. Defaults to
                True.
            bell_threshold (int, float): Duration threshold [hr] below which
                Bell's formula is applied. Default is 1 hr. Force to 0 to
                disable.
            idf_model_type (int, optional): Type of parametric model to fit.
                Only used if parametric=True.
                Options:
                    1: Type I power law: i(t) = a / t^b
                    2: Type II power law: i(t) = a / (t^b + c)
                    3: Type III power law: i(t) = a / (t + c)^b
                Defaults to 2.
            duration_limits (ArrayLike, optional): Min and max duration limits
                in hours for the IDF curve. Defaults to (0, 72).
            **storm_kwargs: Additional parameters depending on storm type.
                - For triangular: Optional peak_time_ratio (float) in hours.
                - For alternating_block: No extra parameters needed.
                - For theoretical design methods: No extra parameters needed.
                - For SciPy distributions: `loc`, `scale`, and shape params.
        """
        # Avoid mutable default
        if storm_kwargs is None:
            storm_kwargs = {}

        # Check design computation path
        if storm_type in self.PREDEFINED_STORMS:
            self.pr_dimless = self._predefined_hyetograph(storm_type)
            self._pr_dimless_cum = self.pr_dimless.cumsum()
        elif hasattr(st, storm_type):
            self.pr_dimless = self._scipy_hyetograph(storm_type,
                                                     **storm_kwargs)
            self._pr_dimless_cum = self.pr_dimless.cumsum()
        elif storm_type in ['triangular', 'alternating_block',
                            'instant_intensity']:
            if idf_scaling is False:
                raise ValueError(
                    "idf_scaling must be True for design storm methods.")
            # Design storm methods doesnt rely on dimensionless hyetographs
            self.pr_dimless = None
            self._pr_dimless_cum = None
            if storm_type == 'instant_intensity':
                raise NotImplementedError(
                    f"{storm_type} method not yet implemented.")
        else:
            raise ValueError(f"Unknown storm_type '{storm_type}'")

        # Determine if IDF scaling is to be applied
        if rain24 is not None:
            self.idf_scaling = idf_scaling
        else:
            self.idf_scaling = True

        # Initialize base IDF class
        super().__init__(timestep=timestep, rain24=rain24, idf=idf,
                         duration_limits=duration_limits)

        # Store input arguments
        self.storm_type = storm_type
        self.storm_kwargs = storm_kwargs

        # Initialize additional attributes
        self.time = None
        self.pr = None
        self.pr_cum = None
        self.pr_depth = None
        self.storm_duration = None
        self.storm_onset_time = None

    def copy(self) -> Type['RainStorm']:
        """
        Create a deep copy of the class itself
        """
        return pycopy.deepcopy(self)

# --------------------------------- Utilities -------------------------------- #
    def _build_time_axes(self, total_duration: float, tail_duration: float
                         ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build base and extended time arrays.
        """
        dt_eps = self.timestep * 1e-3
        base_time = np.arange(0, total_duration + dt_eps, self.timestep)
        extended_time = np.arange(0, total_duration + tail_duration + dt_eps,
                                  self.timestep)
        base_time = xr.DataArray(base_time, dims=['time'],
                                 coords={'time': base_time})
        extended_time = xr.DataArray(extended_time, dims=['time'],
                                     coords={'time': extended_time})
        return base_time, extended_time

    def _shift_timeseries_1d(self, arr: np.ndarray,
                             shift_hours: float) -> np.ndarray:
        """Shift cumulative time series forward by prepending zeros."""
        if shift_hours <= 0:
            return arr
        shift_steps = int(np.round(shift_hours / self.timestep))
        return np.concatenate([np.zeros(shift_steps), arr])[:len(arr)]

    def _apply_onset_time(self, da: xr.DataArray,
                          xr_onset_time: xr.DataArray) -> xr.DataArray:
        """Apply onset time delay to da"""
        if (xr_onset_time == 0).all():
            return da
        da_shifted = xr.apply_ufunc(self._shift_timeseries_1d, da,
                                    xr_onset_time, input_core_dims=[
                                        ['time'], []],
                                    output_core_dims=[['time']],
                                    vectorize=True)
        return da_shifted.transpose(*da.dims)

# -------------------- IDF functionality and design storms ------------------- #
    def fit(self, parametric: bool = True, idf_model_type: int = 2,
            bell_threshold: float = 1, **kwargs) -> Type['RainStorm']:
        """
        Rewrite the fit method to include hyetograph generation along with IDF
        fitting/scaling.

        Args:
            parametric (bool): If True, fits a parametric model to the IDF
                curve. If False, interpolate/extrapolate the sampled IDF curve
                directly. Defaults to True.
            idf_model_type (int, optional): Type of parametric model to fit.
                Options:
                    1: Type I power law: i(t) = a / t^b
                    2: Type II power law: i(t) = a / (t^b + c)
                    3: Type III power law: i(t) = a / (t + c)^b
                Defaults to 2.
            bell_threshold (int, float): Duration threshold [hr] below which
                Bell's formula is applied. Default is 1 hr. Force to 0 to
                disable.
            **kwargs: Additional keyword arguments for duration_coef if rain24
                is provided (e.g., ref_pgauge, expon, etc.).
        """
        if self.idf_scaling:
            super().fit(parametric=parametric,
                        idf_model_type=idf_model_type,
                        bell_threshold=bell_threshold,
                        **kwargs)
        else:
            # Build IDF without scaling
            self.rdf = self.rain24.expand_dims({'duration': self.durations})
            self.idf = self._rdf2idf(self.rdf)
            self.dcoefs = self._rdf2dcoef(self.rdf)
        return self

# -------------------- Synthetic Hyetograph functionality -------------------- #
    def _predefined_hyetograph(self, kind: str) -> pd.Series:
        """
        Synthetic hyetograph generator function for predefined synthetic
        hyetographs.

        Args:
            kind (str): Type of synthetic hyetograph to use.
                Can be any of:
                   > "SCS_X" with X = I24,IA24,II6,II12,II24,II48,III24
                   > "GX_Benitez1985" with X = 1,2,3
                   > "GX_Espildora1979" with X = 1,2,3
                   > "GXpY_Varas1985" with X = 1,2,3,4 and Y=10,25,50,75,90
        """
        return SHYETO_DATA[kind]

    def _scipy_hyetograph(self, kind: str, loc: float = 0.5, scale: float = 0.1,
                          flip: bool = False, n: int = 1000, **kwargs: Any
                          ) -> pd.Series:
        """Synthetic hyetograph generator function for any of scipy
        distributions. The synthetic hyetograph will be built with the given
        loc, scale and scipy default parameters.
        """
        time_dimless = np.linspace(0, 1, n)
        distr = getattr(st, kind)
        shyeto = distr.pdf(time_dimless, loc=loc, scale=scale, **kwargs)
        shyeto /= np.sum(shyeto)  # Normalize to sum 1
        if flip:
            shyeto = shyeto[::-1]
        return pd.Series(shyeto, index=time_dimless)

    def _dimensionless_shyeto(self, base_time: np.ndarray,
                              xr_duration: xr.DataArray,
                              **interp_kwargs) -> xr.DataArray:
        """
        Interpolate cached dimensionless cumulative hyetograph over
        normalized time per cell.
        """
        source_cum = xr.DataArray(
            self._pr_dimless_cum.values,
            dims=['source_time'],
            coords={'source_time': self.pr_dimless.index.values}
        )
        norm_time = (base_time / xr_duration).clip(0, 1)
        shyeto = source_cum.interp(source_time=norm_time, **interp_kwargs)

        # Normalize final value to 1.0 to ensure pr_cum matches rainfall
        shyeto = shyeto / shyeto.isel(time=-1, drop=False)
        shyeto = shyeto.assign_coords(time=base_time)
        shyeto['time'].attrs = {'standard_name': 'time', 'units': 'hr'}
        return shyeto

# ----------------------------------- Core ----------------------------------- #
    def _build_inputs(self,
                      target_duration: ArrayLike,
                      onset_time: ArrayLike,
                      ) -> tuple[xr.DataArray, xr.DataArray,
                                 xr.DataArray, xr.DataArray]:
        """
        Broadcast target_duration/onset_time to the rdf curve and validate
        input values.
        """
        # Validate and transform inputs
        if self.rdf is None:
            raise ValueError("IDF curve must be computed before building a "
                             "storm. Call fit() method first.")

        xr_rdf = self.rdf.copy()
        xr_target_duration = obj2xarray(target_duration,
                                        name='target_duration')
        xr_onset_time = obj2xarray(onset_time,
                                   name='onset_time')

        for name, arr in [('target_duration', xr_target_duration),
                          ('onset_time', xr_onset_time)]:
            if arr is not None and (arr < 0).any():
                raise ValueError(f"{name} must be non-negative")

        # Rename generic dimension names to avoid broadcasting conflicts
        arrays = {'target_duration': xr_target_duration,
                  'onset_time': xr_onset_time}
        for prefix, da in arrays.items():
            if da is not None and 'dim_' in str(da.dims):
                new_dims = {dim: f'{prefix}_{i}'
                            for i, dim in enumerate(da.dims)
                            if dim.startswith('dim_')}
                arrays[prefix] = da.rename(new_dims)
        xr_target_duration, xr_onset_time = (arrays['target_duration'],
                                             arrays['onset_time'])

        # Broadcast to same shape
        xr_target_duration, _ = xr.broadcast(xr_target_duration,
                                             xr_rdf.isel(duration=0))
        xr_onset_time, _ = xr.broadcast(xr_onset_time, xr_target_duration)
        return (xr_target_duration, xr_onset_time)

    def _evaluate_rdf(self, xr_duration: xr.DataArray, **interp_kwargs
                      ) -> xr.DataArray:
        """Evaluate RDF curve at target durations."""
        dmin, dmax = self.duration_limits
        if (xr_duration < dmin).any() or (xr_duration > dmax).any():
            raise ValueError(
                f"Cannot compute storm for durations outside IDF curve bounds "
                f"[{dmin}, {dmax}] hr. Requested: [{float(xr_duration.min())}, "
                f"{float(xr_duration.max())}] hr")
        rain = self.rdf.interp(duration=xr_duration, **interp_kwargs)
        rain = rain.reset_coords(drop=True)
        return rain

    def _build_triangular(self, base_time: np.ndarray,
                          xr_duration: xr.DataArray, peak_time_ratio: float,
                          **interp_kwargs) -> tuple[xr.DataArray,
                                                    xr.DataArray]:
        """
        Generate cumulative and incremental rainfall using the triangular
        hyetograph method.
        """
        if peak_time_ratio is None:
            peak_time_ratio = 0.5  # Default to symmetric triangle
        if not (0 < peak_time_ratio < 1):
            raise ValueError(
                "peak_time_ratio must be between 0 and 1 (dimensionless)")
        pp = self._evaluate_rdf(xr_duration, **interp_kwargs)
        prate = 2 * pp / xr_duration
        ptime = abs(xr_duration * peak_time_ratio - base_time).idxmin('time')

        # Build hyetograph
        pr = xr.full_like(prate, np.nan).copy().expand_dims(
            {'time': base_time}).copy()
        pr.loc[{'time': ptime}] = prate
        pr.loc[{'time': 0}] = 0
        pr.loc[{'time': xr_duration}] = 0
        pr = pr.interpolate_na(dim='time', method='linear').fillna(0)
        pr_depth = pr * self.timestep
        pr_cum = pr_depth.cumsum('time')
        return pr_cum, pr_depth, pr

    def _build_altblock(self, base_time: np.ndarray,
                        xr_duration: xr.DataArray,
                        **interp_kwargs) -> tuple[xr.DataArray,
                                                  xr.DataArray]:
        """
        Generate cumulative and incremental rainfall using the alternating
        block method.
        """
        dims = {dim: coord for dim, coord in xr_duration.coords.items()}
        time = base_time.expand_dims(dims)
        time = time.transpose('time', *dims.keys())
        time = time.clip(0, xr_duration)
        pr_cum = self._evaluate_rdf(time, **interp_kwargs)
        pr_depth = pr_cum.diff('time').pad(time=(1, 0), constant_values=0)
        pr = pr_depth / self.timestep

        pr = xr.apply_ufunc(alternating_block_sort,
                            pr.where(pr > 0),
                            input_core_dims=[['time']],
                            output_core_dims=[['time']],
                            vectorize=True).transpose('time', *dims.keys())
        pr = pr.pad(time=(1, 0), constant_values=0)
        pr = pr.fillna(0).isel(time=slice(None, -1))
        pr_depth = pr * self.timestep
        pr_cum = pr_depth.cumsum('time')
        return pr_cum, pr_depth, pr

    def _build_shyeto(self, base_time: np.ndarray,
                      xr_duration: xr.DataArray,
                      **interp_kwargs) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Build cumulative and incremental rainfall based on dimensionless
        hyetograph scaling.
        """
        shyeto = self._dimensionless_shyeto(base_time,
                                            xr_duration,
                                            **interp_kwargs)
        xr_rainfall = self._evaluate_rdf(xr_duration, **interp_kwargs)
        pr_cum = shyeto * xr_rainfall

        # Differentiate to get precipitation rate
        pr_depth = pr_cum.diff('time').pad(time=(1, 0), constant_values=0)
        pr = pr_depth / self.timestep
        return pr_cum, pr_depth, pr

    def compute(self,
                target_duration: int | float | ArrayLike,
                onset_time: int | float | ArrayLike = 0,
                tail_duration: float = 0,
                interp_kwargs: dict = {'method': 'linear'}
                ) -> Type['RainStorm']:
        """
        Trigger computation of design storm for a given storm duration
        and total precipitation.

        Args:
            target_duration (float or array_like): Storm duration in hours.
                Can be scalar or broadcastable to rainfall/idf_curve (e.g.,
                spatial grid with same dims).
            onset_time (float or array_like, optional): Time delay in hours for
                storm onset. If scalar, shifts all precipitation uniformly.
                If array_like, allows spatially variable timing. Default is 0.
            tail_duration (float, optional): Additional time in hours to extend
                the time axis after the storm ends. Used to pad with zeros after
                precipitation. Default is 0.
            interp_kwargs (dict): Extra arguments for time interpolation.
                Defaults to {'method': 'linear'}.

        Returns:
            Updated class instance with computed storm.
        """
        inputs = self._build_inputs(target_duration, onset_time)
        (xr_target_duration, xr_onset_time) = inputs

        # Build time axes
        total_duration = xr_target_duration.max().item()
        total_duration += float(xr_onset_time.max().item())
        base_time, extended_time = self._build_time_axes(total_duration,
                                                         tail_duration)

        # Compute cumulative and incremental rainfall
        if self.storm_type in ['instant_intensity']:
            raise NotImplementedError(
                f"{self.storm_type} method not yet implemented.")
        elif self.storm_type == 'triangular':
            r = self.storm_kwargs.get('peak_time_ratio')
            pr_cum, pr_depth, pr = self._build_triangular(base_time,
                                                          xr_target_duration,
                                                          peak_time_ratio=r,
                                                          **interp_kwargs)
        elif self.storm_type == 'alternating_block':
            pr_cum, pr_depth, pr = self._build_altblock(base_time,
                                                        xr_target_duration,
                                                        **interp_kwargs)
        else:
            pr_cum, pr_depth, pr = self._build_shyeto(base_time,
                                                      xr_target_duration,
                                                      **interp_kwargs)

        # Apply onset time shifts
        pr_cum = self._apply_onset_time(pr_cum, xr_onset_time)
        pr_depth = self._apply_onset_time(pr_depth, xr_onset_time)
        pr = self._apply_onset_time(pr, xr_onset_time)

        # Reindex to extended time axis and fill NaNs with 0
        # pr = pr.reindex({'time': extended_time}).fillna(0)
        # pr_depth = pr_depth.reindex({'time': extended_time}).fillna(0)
        # pr_cum = pr_cum.reindex({'time': extended_time}, method='pad')

        # Assign names and attributes
        pr.name, pr.attrs = 'pr', {
            'standard_name': 'precipitation rate', 'units': 'mm/hr'}

        pr_cum.name, pr_cum.attrs = 'pr_cum', {
            'standard_name': 'cumulative precipitation', 'units': 'mm'}

        pr_depth.name, pr_depth.attrs = 'pr_depth', {
            'standard_name': 'precipitation depth', 'units': 'mm'}

        # bias = pr.integrate('time') + pr_cum.isel(time=0, drop=True)
        # bias = abs(bias-self.rain24).max()
        # if bias > 1e-3:
        #     warnings.warn(
        #         f"Mass balance error detected in storm computation. "
        #         f"Max bias: {bias:.4f} mm. Check input parameters.",
        #         UserWarning
        #     )

        # Update cached class attributes
        self.time = pr.coords['time']
        self.pr = pr
        self.pr_cum = pr_cum
        self.pr_depth = pr_depth
        self.storm_duration = target_duration
        self.storm_onset_time = onset_time
        return self
