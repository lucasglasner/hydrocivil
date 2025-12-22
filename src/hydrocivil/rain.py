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
from .misc import obj2xarray, series2func, rsquared
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
    which is generally valid for cyclonic precipitation on flat terrain.
    However, for convective rainfall or rainfall on complex terrain, a
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


def alternating_block_sort(arr: ArrayLike, axis=0):
    """
    Sorts an array in an alternating block pattern along a specified axis.

    If the minimum value is 0, it will only be placed at the beginning,
    never at the end of the array.

    Args:
        arr (ArrayLike): Input array to be rearranged.
        axis (int, optional): Axis along which to sort and rearrange.
            Default is 0.
    Returns
        rearranged (ArrayLike): Array with values rearranged in alternating
            block pattern, with the highest values placed near the center.
    """
    arr = np.asarray(arr)

    # Sort in descending order. Get size, create index and empty arrays
    arr_sorted = np.sort(arr, axis=axis)
    arr_sorted = np.flip(arr_sorted, axis=axis)

    n = arr.shape[axis]
    rearranged = np.zeros_like(arr_sorted)
    indices = [slice(None)] * arr.ndim

    # Fill the rearranged array with alternating block pattern
    for i in range(n):
        pos = n // 2 + (-1) ** i * ((i + 1) // 2)

        # Select the i-th slice and place it at position 'pos'
        indices[axis] = i
        source_slice = arr_sorted[tuple(indices)]

        indices[axis] = pos
        rearranged[tuple(indices)] = source_slice

    # Check if minimum value is 0 and if it ended up at the end,
    # move it to the beginning
    indices_end = [slice(None)] * arr.ndim
    indices_end[axis] = -1
    indices_start = [slice(None)] * arr.ndim
    indices_start[axis] = 0

    if np.min(arr) == 0 and np.min(rearranged[tuple(indices_end)]) == 0:
        # Swap the last element with the first element
        temp = rearranged[tuple(indices_start)].copy()
        rearranged[tuple(indices_start)] = rearranged[tuple(indices_end)]
        rearranged[tuple(indices_end)] = temp

    return rearranged

# ----------------------------- IDF Relationships ---------------------------- #


class IDF:
    def __init__(self,
                 timestep: float,
                 rain: ArrayLike = None,
                 idf: ArrayLike = None,
                 bell_threshold: float = 1,
                 model_type: int = 2,
                 duration_limits: ArrayLike = (0, 72),
                 **kwargs) -> None:
        """
        Intensity-Duration-Frequency (IDF) relationship handler.

        Args:
            timestep (float): Time step or resolution in hours.
            rain (ArrayLike, optional): Rainfall amount(s) in mm per 24 hours.
            idf (ArrayLike, optional): IDF curve as an array with an
                identifiable 'duration' dimension. If None, IDF will be computed
                from rainfall using duration_coef. Defaults to None.
            bell_threshold (int, float): Duration threshold [hr] below which
                Bell's formula is applied. Default is 1 hr.
                Force to 0 to disable.
            model_type (int, optional): Type of parametric model to fit.
                Options:
                    1: Type I power law: i(t) = a / t^b
                    2: Type II power law: i(t) = a / (t^b + c)
                    3: Type III power law: i(t) = a / (t + c)^b
                Defaults to 2.
            duration_limits (ArrayLike, optional): Min and max duration limits
                in hours for the IDF curve. Defaults to (0, 72).
            **kwargs: Additional keyword arguments for
                duration_coef if rain is provided.
        """
        # Validate inputs
        self._validate_inputs(timestep, rain, idf, model_type, duration_limits)
        if rain is None and idf is not None:
            self.computation_path = 'from_idf'
        else:
            self.computation_path = 'from_rain'

        # Convert rain and idf to xarray DataArrays
        xr_rain = obj2xarray(rain, name='rain') if rain is not None else None
        xr_idf = obj2xarray(idf, name='idf') if idf is not None else None

        # Store input arguments
        self.timestep = timestep
        self.rain = xr_rain
        self.idf = None
        self.rdf = None
        self.dcoefs = None
        self.model_type = model_type
        self.duration_limits = duration_limits
        self.kwargs = kwargs
        self.bell_threshold = bell_threshold

        # Build aditional attributes
        self.durations = self._build_duration_axis()
        self.idf_sampled = xr_idf
        self.rdf_sampled = None
        self.dcoefs_sampled = None
        self.rsquared = None
        self.fitcoefs_ = None

    # -------------------------------- Validations --------------------------- #

    def _validate_inputs(self,
                         timestep: float,
                         rain: ArrayLike,
                         idf: ArrayLike,
                         model_type: int,
                         duration_limits: ArrayLike):
        """
        Validate input parameters for IDF class.
        """
        if (rain is not None) == (idf is not None):
            raise ValueError("Must specify exactly one of 'rain' or 'idf'")

        if timestep <= 0:
            raise ValueError(f"timestep must be positive, got {timestep}")

        if model_type not in [1, 2, 3]:
            raise ValueError("model_type must be 1, 2, or 3")

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
        xr_rdf.name = 'rain'
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
    def _powerlaw_type1(t, b, rain24):
        """
        Type I power law model with constraint i(24) = rain24/24
        i(t) = a / t^b, where a = rain24 * 24^(b-1)
        """
        a = rain24 * (24 ** (b - 1))
        return a / (t ** b)

    @staticmethod
    def _powerlaw_type2(t, b, c, rain24):
        """
        Type II power law model with constraint i(24) = rain24/24
        i(t) = a / (t^b + c), where a = rain24 * (24^b + c) / 24
        """
        a = rain24 * (24 ** b + c) / 24
        return a / (t ** b + c)

    @staticmethod
    def _powerlaw_type3(t, b, c, rain24):
        """
        Type III power law model with constraint i(24) = rain24/24
        i(t) = a / (t + c)^b, where a = rain24 * (24 + c)^b / 24
        """
        a = rain24 * ((24 + c) ** b) / 24
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

    def _sample_idf_curve(self, durations: ArrayLike = None) -> xr.DataArray:
        """
        Build the IDF and RDF curve based on rain and duration coeficients
        """
        # Build
        if durations is None:
            durations = DURCOEFS_PGAUGES.index.values
        durations = xr.DataArray(durations, dims=['duration'])
        cds = duration_coef(durations, **self.kwargs)
        cds = xr.DataArray(cds, dims=['duration'], name='cds',
                           coords={'duration': durations})
        rdf = self.rain.expand_dims({'duration': durations}, axis=0) * cds
        idf = rdf / rdf['duration']

        # Name and attrs
        rdf.name = 'rain'
        rdf.attrs = {'standard_name': 'total_rainfall_vs_duration',
                     'units': 'mm'}

        idf.name = 'intensity'
        idf.attrs = {'standard_name': 'intensity_vs_duration',
                     'units': 'mm/hr'}

        self.idf_sampled = idf
        self.rdf_sampled = rdf
        self.dcoefs_sampled = self._rdf2dcoef(rdf)
        return idf, rdf

    def _fit_parametric_model(self, **kwargs):
        """
        Fit a parametric IDF model to the sampled IDF curve and store the
        fit coefficients and R-squared value.

        The fit enforces the constraint i(24) = rain/24, reducing the number
        of free parameters by one.
        """
        func_map = {
            1: (self._powerlaw_type1, [[], []]),      # Only b parameter + r2
            2: (self._powerlaw_type2, [[], [], []]),  # b, c parameters + r2
            3: (self._powerlaw_type3, [[], [], []])   # b, c parameters + r2
        }

        func, output_dims = func_map[self.model_type]

        def _func2fit(i, t, rain24, **kwargs):
            # Create wrapper with explicit parameters based on model type
            if self.model_type == 1:
                # Type I: only b parameter
                def func_with_rain24(t_fit, b): return func(t_fit, b, rain24)
            else:
                # Type II and III: b, c parameters
                def func_with_rain24(t_fit, b, c): return func(
                    t_fit, b, c, rain24)

            popt, _ = curve_fit(func_with_rain24, t, i,
                                bounds=(0, np.inf), **kwargs)
            return (*popt, rsquared(i, func(t, *popt, rain24)))

        # Get rain at 24 hours
        rain24 = self.idf_sampled.interp(duration=24) * 24

        output = xr.apply_ufunc(_func2fit,
                                self.idf_sampled,
                                self.idf_sampled['duration'],
                                rain24,
                                input_core_dims=[
                                    ['duration'], ['duration'], []],
                                output_core_dims=output_dims,
                                vectorize=True,
                                kwargs=kwargs)
        popt = output[:-1]
        r2 = output[-1]
        self.rsquared = r2
        self.fitcoefs_ = popt
        return popt, r2

    def _force_bell_threshold(self, idf: xr.DataArray) -> xr.DataArray:
        """
        Force Bell threshold application on the IDF curve.
        """
        if self.bell_threshold <= 0:
            return idf
        name, attrs = idf.name, idf.attrs
        rdf = self._idf2rdf(idf)
        dcoef = self._rdf2dcoef(rdf)

        bell_mask = idf['duration'] < self.bell_threshold
        if bell_mask.any():
            t, cd1 = xr.broadcast(idf['duration'][bell_mask],
                                  dcoef.interp(duration=1))
            dcoef_bell = bell_coef(t, cd1)
            dcoef.loc[{'duration': idf['duration'][bell_mask]}] = dcoef_bell
            idf = dcoef * rdf.interp(duration=24) / idf['duration']
            idf.name = name
            idf.attrs = attrs
        return idf

    def _evaluate_parametric_model(self, durations: ArrayLike, model_type: int
                                   ) -> xr.DataArray:
        """
        Evaluate the fitted parametric model at given durations.
        """
        durations = xr.DataArray(durations, dims=['duration'])
        if self.fitcoefs_ is None and self.rsquared is None:
            raise ValueError("Model must be fitted before evaluation.")

        model_map = {
            1: self._powerlaw_type1,
            2: self._powerlaw_type2,
            3: self._powerlaw_type3
        }

        model = model_map[model_type]

        # Get rain for a 24 hour storm
        rain24 = self.idf_sampled.interp(duration=24) * 24

        # Evaluate model with rain24 constraint
        idf = model(durations, *self.fitcoefs_, rain24)
        idf = idf.assign_coords({'duration': durations})
        idf = idf.transpose(*self.idf_sampled.dims)
        idf.name = 'intensity'
        idf.attrs = {'standard_name': 'intensity_vs_duration',
                     'units': 'mm/hr'}
        return idf

    def fit(self, parametric=False) -> Type['IDF']:
        """
        Fit the Intensity-Duration curve using the specified computation path.
        This method fits an IDF curve based on either rainfall data or existing
        IDF data, applies a parametric model, and computes related rainfall
        properties including RDF(Rainfall-Duration) and disaggregation/duration
        coefficients.

        Args:
            parametric (bool): If True, fits a parametric model to the IDF
                curve. If False, interpolate/extrapolate the sampled IDF curve
                directly. Defaults to False.

        Returns:
            IDF: The instance itself with updated attributes.
        Raises:
            AttributeError: If required attributes for the specified
                computation_path are not properly initialized in the class
                constructor.
        """
        if self.computation_path == 'from_rain':
            kwargs = self.kwargs.copy()
            if 'ref_pgauge' in kwargs and kwargs['ref_pgauge'] != 'Grunsky':
                if parametric:
                    self._sample_idf_curve()
                    self._fit_parametric_model()
                    idf = self._evaluate_parametric_model(self.durations,
                                                          self.model_type)
                else:
                    self._sample_idf_curve(self.durations)
                    idf = self.idf_sampled
                idf = self._force_bell_threshold(idf)
                self.idf = idf
                self.rdf = self._idf2rdf(self.idf)
                self.dcoefs = self._idf2dcoef(self.idf)
            else:
                self._sample_idf_curve(self.durations)
                self.idf = self.idf_sampled
                self.rdf = self.rdf_sampled
                self.dcoefs = self.dcoefs_sampled

        if self.computation_path == 'from_idf':
            self.rdf_sampled = self._idf2rdf(self.idf_sampled)
            self.dcoefs_sampled = self._idf2dcoef(self.idf_sampled)
            if parametric:
                self._fit_parametric_model()
                idf = self._evaluate_parametric_model(self.durations,
                                                      self.model_type)
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
            idf = self._force_bell_threshold(idf)
            self.idf = idf
            self.rdf = self._idf2rdf(self.idf)
            self.dcoefs = self._idf2dcoef(self.idf)

        self._force0duration()
        return self


class RainStorm:
    pass

# class RainStorm:
#     """
#     The class can be used to build rainstorms that follow any of scipy
#     theoretical distributions (e.g 'norm', 'skewnorm', 'gamma', etc) or
#     the empirical rain distributions of the SCS type I, IA, II, III and the
#     Chilean synthetic hyetographs of (Espildora and Echavarría 1979),
#     (Benitez and Verni 1985) and (Varas 1985).

#     The class also supports the 'alternating_block' method, which generates
#     hyetographs directly from IDF/RDF curves using the alternating block
#     algorithm (Chow, 1985).
#     """
#     PREDEFINED_STORMS = SHYETO_DATA.columns

#     def __init__(self, kind: str, timestep: float, **kwargs: Any) -> None:
#         """
#         Synthetic RainStorm builder

#         Args:
#             kind (str): Type of storm model to use.
#                 - Options:
#                    G1_Espildora1979, G2_Espildora1979, G3_Espildora1979,
#                    G1_Benitez1985, G2_Benitez1985, G3_Benitez1985,
#                    SCS_I24, SCS_IA24,
#                    SCS_II6, SCS_II12, SCS_II24, SCS_II48,
#                    SCS_III24,
#                    G1p10_Varas1985, G1p25_Varas1985, G1p50_Varas1985,
#                    G1p75_Varas1985, G1p90_Varas1985,
#                    G2p10_Varas1985, G2p25_Varas1985, G2p50_Varas1985,
#                    G2p75_Varas1985, G2p90_Varas1985,
#                    G3p10_Varas1985, G3p25_Varas1985, G3p50_Varas1985,
#                    G3p75_Varas1985, G3p90_Varas1985,
#                    G4p10_Varas1985, G4p25_Varas1985, G4p50_Varas1985,
#                    G4p75_Varas1985, G4p90_Varas1985,
#                 - SciPy distribution (e.g., 'norm', 'gamma')
#                 - alternating_block
#             timestep (float): Storm timestep or resolution in hours.
#             **kwargs: Additional parameters depending on the storm type.
#                 - For predefined storms: No extra parameters needed.
#                 - For SciPy distributions: `loc`, `scale`, and shape params.

#         Examples:
#             RainStorm('SCS_I24', timestep=0.5)
#             RainStorm('G2_Benitez1985', timestep=0.5)
#             RainStorm('G3_Espildora1979', timestep=0.5)
#             RainStorm('G4p10_Varas1985', timestep=0.5)
#             RainStorm('norm', timestep=0.5, loc=0.5, scale=0.2)
#             RainStorm('gamma', timestep=0.5, loc=0, scale=0.15, a=2)
#             RainStorm('alternating_block', timestep=0.5)
#         """
#         if timestep <= 0:
#             raise ValueError(f"timestep must be positive, got {timestep}")

#         self.kind = kind
#         self.timestep = timestep
#         self.duration = self.rainfall = self.time = None
#         self.idf_curve = self.rdf_curve = None
#         self.pr = self.pr_cum = None

#         if kind == 'alternating_block':
#             # Alternating block method doesn't use dimensionless hyetograph
#             self.pr_dimless = None
#             self._pr_dimless_cum = None
#         elif kind in self.PREDEFINED_STORMS:
#             self.pr_dimless = self._predefined_hyetograph(kind)
#             self._pr_dimless_cum = self.pr_dimless.cumsum()
#         elif hasattr(st, kind):
#             self.pr_dimless = self._scipy_hyetograph(kind, **kwargs)
#             self._pr_dimless_cum = self.pr_dimless.cumsum()
#         else:
#             raise ValueError(f"Unknown storm type: {kind}")

#     def copy(self) -> Type['RainStorm']:
#         """
#         Create a deep copy of the class itself
#         """
#         return pycopy.deepcopy(self)

#     def _build_inputs(self,
#                       duration: ArrayLike,
#                       rainfall: ArrayLike,
#                       idf_curve: ArrayLike,
#                       onset_time: ArrayLike,
#                       ) -> tuple[xr.DataArray, xr.DataArray,
#                                  xr.DataArray, xr.DataArray]:
#         """
#         Broadcast duration/rainfall/onset_time/idf_curve, validate values,
#         return max total duration.
#         """
#         xr_duration = obj2xarray(duration)
#         xr_rainfall = obj2xarray(rainfall) if rainfall is not None else None
#         xr_idf_curve = obj2xarray(idf_curve) if idf_curve is not None else None
#         xr_onset_time = obj2xarray(onset_time)

#         # Rename generic dimension names to avoid broadcasting conflicts
#         arrays = {'duration': xr_duration, 'rainfall': xr_rainfall,
#                   'idf_curve': xr_idf_curve, 'onset_time': xr_onset_time}
#         for prefix, da in arrays.items():
#             if da is not None and 'dim_' in str(da.dims):
#                 new_dims = {dim: f'{prefix}_{i}'
#                             for i, dim in enumerate(da.dims)
#                             if dim.startswith('dim_')}
#                 arrays[prefix] = da.rename(new_dims)
#         xr_duration, xr_rainfall, xr_idf_curve, xr_onset_time = (
#             arrays['duration'], arrays['rainfall'],
#             arrays['idf_curve'], arrays['onset_time'])

#         # Broadcast to same shape, avoiding idf_curve's 'duration' dim
#         if xr_idf_curve is not None and 'duration' in xr_idf_curve.dims:
#             # Broadcast xr_duration against idf_curve's other dims
#             # (excluding curve)
#             idf_other_dims = xr_idf_curve
#             # Create a view with other dims only by taking a single
#             # slice along the curve dimension; this preserves other dims
#             # for broadcast
#             idf_other_dims = idf_other_dims.isel(
#                 duration=0, drop=True)
#             xr_duration, _ = xr.broadcast(xr_duration, idf_other_dims)
#             xr_duration, xr_onset_time = xr.broadcast(xr_duration,
#                                                       xr_onset_time)
#         elif xr_rainfall is not None:
#             xr_duration, xr_rainfall = xr.broadcast(xr_duration,
#                                                     xr_rainfall)
#             xr_duration, xr_onset_time = xr.broadcast(xr_duration,
#                                                       xr_onset_time)
#         else:
#             xr_duration, xr_onset_time = xr.broadcast(xr_duration,
#                                                       xr_onset_time)

#         # Validate inputs
#         for name, arr in [('duration', xr_duration),
#                           ('rainfall', xr_rainfall),
#                           ('idf_curve', xr_idf_curve),
#                           ('onset_time', xr_onset_time)]:
#             if arr is not None and (arr < 0).any():
#                 raise ValueError(f"{name} must be non-negative")

#         return (xr_duration, xr_rainfall, xr_idf_curve, xr_onset_time)

#     def _build_time_axes(self, total_duration: float, tail_duration: float
#                          ) -> tuple[np.ndarray, np.ndarray]:
#         """
#         Build base and extended time arrays.
#         """
#         dt_eps = self.timestep * 1e-3
#         base_time = np.arange(0, total_duration + dt_eps, self.timestep)
#         extended_time = np.arange(0, total_duration + tail_duration + dt_eps,
#                                   self.timestep)
#         return base_time, extended_time

#     def _validate_idf_curve(self, xr_idf_curve: xr.DataArray) -> xr.DataArray:
#         """
#         Validate IDF curve structure and sort by duration if needed.
#         """
#         if 'duration' not in xr_idf_curve.dims:
#             raise ValueError("idf_curve must have a 'duration' dimension")
#         if (xr_idf_curve < 0).any():
#             raise ValueError("IDF curve values must be non-negative")
#         # Ensure monotonic increasing duration coordinate
#         if not xr_idf_curve['duration'].to_series().is_monotonic_increasing:
#             xr_idf_curve = xr_idf_curve.sortby('duration')
#         return xr_idf_curve

#     def _apply_idf_scaling(self, xr_duration: xr.DataArray,
#                            xr_rainfall: xr.DataArray,
#                            idf_kwargs: dict) -> xr.DataArray:
#         """Scale rainfall to target durations using duration_coef."""
#         return xr_rainfall * duration_coef(xr_duration, **idf_kwargs)

#     def _build_rdf_curve_from_rainfall(self, xr_rainfall: xr.DataArray,
#                                        idf_scaling: bool, idf_kwargs: dict
#                                        ) -> xr.DataArray:
#         """Build a rainfall-duration curve from a rainfall value."""
#         durations_rdf = np.arange(0, 72 + self.timestep, self.timestep)
#         rainfall_base = float(
#             xr_rainfall) if xr_rainfall.ndim == 0 else xr_rainfall
#         rainfall_values = (duration_coef(durations_rdf, **idf_kwargs)
#                            if idf_scaling else np.ones_like(durations_rdf))
#         xr_rdf_curve = self._create_rdf_array_from_values(
#             rainfall_values, rainfall_base, durations_rdf)
#         return self._finalize_rdf_curve(xr_rdf_curve)

#     def _create_rdf_array_from_values(self, rainfall_values: np.ndarray,
#                                       rainfall_base: float | xr.DataArray,
#                                       durations: np.ndarray) -> xr.DataArray:
#         """Create xarray.DataArray for RDF curve from rainfall values."""
#         if isinstance(rainfall_base, (float, int)):
#             return xr.DataArray(rainfall_values * rainfall_base,
#                                 dims=['duration'],
#                                 coords={'duration': durations})
#         values_da = xr.DataArray(rainfall_values,
#                                  dims=['duration'],
#                                  coords={'duration': durations})
#         return rainfall_base * values_da

#     def _finalize_rdf_curve(self, xr_rdf_curve: xr.DataArray) -> xr.DataArray:
#         """Finalize RDF curve with boundary conditions and metadata."""
#         xr_rdf_curve.loc[{'duration': 0}] = 0
#         xr_rdf_curve.name = 'rainfall'
#         xr_rdf_curve.attrs = {'standard_name': 'total_rainfall_vs_duration',
#                               'units': 'mm'}
#         return xr_rdf_curve

#     def _idf2rdf(self, xr_rdf: xr.DataArray) -> xr.DataArray:
#         """Convert RDF to IDF curve."""
#         durations = xr_rdf['duration'].values
#         xr_idf = (xr_rdf / durations).where(durations > 0, np.inf)
#         xr_idf.name = 'intensity'
#         xr_idf.attrs = {'standard_name': 'intensity_vs_duration',
#                         'units': 'mm/hr'}
#         return xr_idf

#     def _rdf2idf(self, xr_idf: xr.DataArray) -> xr.DataArray:
#         """Convert IDF to RDF curve."""
#         xr_rdf = xr_idf * xr_idf['duration'].values
#         xr_rdf.loc[{'duration': 0}] = 0
#         xr_rdf.name = 'rainfall'
#         xr_rdf.attrs = {'standard_name': 'total_rainfall_vs_duration',
#                         'units': 'mm'}
#         return xr_rdf

#     def _evaluate_rdf(self, xr_rdf_curve: xr.DataArray,
#                       xr_duration: xr.DataArray,
#                       interp_kwargs: dict) -> xr.DataArray:
#         """Evaluate RDF curve at target durations."""
#         dmin, dmax = float(xr_rdf_curve['duration'].min()), float(
#             xr_rdf_curve['duration'].max())
#         if (xr_duration < dmin).any() or (xr_duration > dmax).any():
#             raise ValueError(
#                 f"Cannot compute storm for durations outside curve bounds "
#                 f"[{dmin}, {dmax}] hr. Requested: [{float(xr_duration.min())}, "
#                 f"{float(xr_duration.max())}] hr")
#         return xr_rdf_curve.interp(duration=xr_duration, **interp_kwargs)

#     def _predefined_hyetograph(self, kind: str) -> pd.Series:
#         """
#         Synthetic hyetograph generator function for predefined synthetic
#         hyetographs.

#         Args:
#             kind (str): Type of synthetic hyetograph to use.
#                 Can be any of:
#                    > "SCS_X" with X = I24,IA24,II6,II12,II24,II48,III24
#                    > "GX_Benitez1985" with X = 1,2,3
#                    > "GX_Espildora1979" with X = 1,2,3
#                    > "GXpY_Varas1985" with X = 1,2,3,4 and Y=10,25,50,75,90
#         """
#         return SHYETO_DATA[kind]

#     def _scipy_hyetograph(self, kind: str, loc: float = 0.5, scale: float = 0.1,
#                           flip: bool = False, n: int = 1000, **kwargs: Any
#                           ) -> pd.Series:
#         """Synthetic hyetograph generator function for any of scipy
#         distributions. The synthetic hyetograph will be built with the given
#         loc, scale and scipy default parameters.
#         """
#         time_dimless = np.linspace(0, 1, n)
#         distr = getattr(st, kind)
#         shyeto = distr.pdf(time_dimless, loc=loc, scale=scale, **kwargs)
#         shyeto /= np.sum(shyeto)  # Normalize to sum 1
#         if flip:
#             shyeto = shyeto[::-1]
#         return pd.Series(shyeto, index=time_dimless)

#     def _dimensionless_shyeto(self, base_time: np.ndarray,
#                               xr_duration: xr.DataArray,
#                               interp_kwargs: dict) -> xr.DataArray:
#         """
#             Interpolate cached dimensionless cumulative hyetograph over
#             normalized time per cell.
#         """
#         source_cum = xr.DataArray(
#             self._pr_dimless_cum.values,
#             dims=['source_time'],
#             coords={'source_time': self.pr_dimless.index.values}
#         )
#         norm_time = (xr.DataArray(base_time, dims=['time']) /
#                      xr_duration).clip(0, 1)
#         shyeto = source_cum.interp(source_time=norm_time, **interp_kwargs)

#         # Normalize final value to 1.0 to ensure pr_cum matches rainfall
#         shyeto = shyeto / shyeto.isel(time=-1, drop=False)
#         shyeto = shyeto.assign_coords(time=base_time)
#         shyeto['time'].attrs = {'standard_name': 'time', 'units': 'hr'}
#         return shyeto

#     def _alternating_block_hyetograph(self,
#                                       xr_rdf_curve: xr.DataArray,
#                                       xr_duration: xr.DataArray,
#                                       base_time: np.ndarray,
#                                       interp_kwargs: dict
#                                       ) -> xr.DataArray:
#         """
#         Generate hyetograph using the alternating block method.

#         Algorithm:
#         1. Interpolate RDF curve at cumulative time points
#         2. Compute rain pulses by differencing cumulative rainfall
#         3. Sort pulses using alternating block pattern
#         4. Return as precipitation rate (mm/hr)

#         Args:
#             xr_rdf_curve: Rainfall-duration curve with 'duration' dimension
#             xr_duration: Target storm durations
#             base_time: Time array for the storm
#             interp_kwargs: Interpolation keyword arguments

#         Returns:
#             Precipitation rate array (mm/hr) with 'time' dimension
#         """
#         if 'duration' not in xr_rdf_curve.dims:
#             raise ValueError(
#                 "alternating_block method requires an RDF curve with "
#                 "'duration' dimension. Use idf_curve parameter or "
#                 "idf_scaling=True."
#             )

#         # Clip base_time to storm duration for each pixel
#         time_da = xr.DataArray(base_time, dims=['time'],
#                                coords={'time': base_time})
#         # Clip time to duration, ensuring we don't exceed storm duration
#         time_clipped = time_da.clip(0, xr_duration)

#         # Interpolate RDF curve at cumulative time points
#         pr_cum_pulses = xr_rdf_curve.interp(duration=time_clipped,
#                                             **interp_kwargs)

#         # Compute rain pulses by differencing
#         # Use pad to prepend the first value, then diff to get pulses
#         pr_cum_padded = pr_cum_pulses.pad(time=(1, 0), constant_values=0)
#         pr_pulses = pr_cum_padded.diff('time')
#         # Assign correct time coordinates
#         pr_pulses = pr_pulses.assign_coords(time=base_time)

#         # Convert to precipitation rate (mm/hr)
#         pr_rate = pr_pulses / self.timestep

#         # Apply alternating block sort along time dimension
#         pr_alternated = xr.apply_ufunc(
#             alternating_block_sort,
#             pr_rate,
#             input_core_dims=[['time']],
#             output_core_dims=[['time']],
#             vectorize=True,
#             kwargs={'axis': 0}
#         )

#         # Restore time coordinate
#         pr_alternated = pr_alternated.assign_coords(time=base_time)
#         pr_alternated['time'].attrs = {'standard_name': 'time',
#                                        'units': 'hr'}

#         # Ensure time dimension is first (xr.apply_ufunc can reorder dims)
#         all_dims = list(pr_alternated.dims)
#         if 'time' in all_dims and all_dims[0] != 'time':
#             all_dims.remove('time')
#             all_dims.insert(0, 'time')
#             pr_alternated = pr_alternated.transpose(*all_dims)

#         return pr_alternated

#     def _shift_timeseries_1d(self, cum_series: np.ndarray,
#                              shift_hours: float) -> np.ndarray:
#         """Shift cumulative time series forward by prepending zeros."""
#         if shift_hours <= 0:
#             return cum_series
#         shift_steps = int(np.round(shift_hours / self.timestep))
#         return np.concatenate([np.zeros(shift_steps),
#                                cum_series])[:len(cum_series)]

#     def _apply_onset_time(self, pr_cum: xr.DataArray,
#                           xr_onset_time: xr.DataArray) -> xr.DataArray:
#         """Apply onset time delay to cumulative precipitation."""
#         if (xr_onset_time == 0).all():
#             return pr_cum
#         pr_cum_shifted = xr.apply_ufunc(self._shift_timeseries_1d, pr_cum,
#                                         xr_onset_time, input_core_dims=[
#                                             ['time'], []],
#                                         output_core_dims=[['time']],
#                                         vectorize=True)
#         return pr_cum_shifted.transpose(*pr_cum.dims)

#     def _build_outputs(self, shyeto: xr.DataArray,
#                        xr_rainfall: xr.DataArray,
#                        xr_onset_time: xr.DataArray,
#                        base_time: np.ndarray,
#                        extended_time: np.ndarray
#                        ) -> tuple[xr.DataArray, xr.DataArray]:
#         """
#         Create cumulative and rate precipitation fields, apply time
#         shift, pad zeros beyond each duration.
#         """
#         pr_cum = shyeto * xr_rainfall
#         # Ensure 'time' is the leading dimension regardless of how
#         # inputs broadcast
#         other_dims = [d for d in pr_cum.dims if d != 'time']
#         pr_cum = pr_cum.transpose(*(['time'] + other_dims))

#         # Apply onset time
#         pr_cum = self._apply_onset_time(pr_cum, xr_onset_time)

#         pr = (pr_cum.diff('time').reindex({'time': base_time}) / self.timestep)
#         pr = pr.reindex({'time': extended_time}).fillna(0)
#         pr.name, pr.attrs = 'pr', {
#             'standard_name': 'precipitation rate', 'units': 'mm/hr'}

#         pr_cum = pr_cum.reindex({'time': extended_time}, method='pad')
#         pr_cum.name, pr_cum.attrs = 'pr_cum', {
#             'standard_name': 'cumulative precipitation', 'units': 'mm'}
#         return pr, pr_cum

#     def _compute_alternating_block_storm(self,
#                                          xr_duration: xr.DataArray,
#                                          xr_onset_time: xr.DataArray,
#                                          base_time: np.ndarray,
#                                          extended_time: np.ndarray,
#                                          interp_kwargs: dict
#                                          ) -> tuple[xr.DataArray, xr.DataArray]:
#         """
#         Compute storm using alternating block method.
#         Returns pr and pr_cum with onset time already applied.
#         """
#         # Generate precipitation rate directly from RDF curve
#         pr = self._alternating_block_hyetograph(
#             self.rdf_curve, xr_duration, base_time, interp_kwargs
#         )

#         # Compute cumulative from rate
#         pr_cum = (pr * self.timestep).cumsum('time')

#         # Get expected total rainfall from RDF curve at target duration
#         expected_rainfall = self.rdf_curve.interp(
#             duration=xr_duration, **interp_kwargs)

#         # Normalize to ensure exact mass conservation
#         # Avoid division by zero for cases where there's no rainfall
#         final_cum = pr_cum.isel(time=-1, drop=True)
#         scale_factor = expected_rainfall / final_cum.where(final_cum != 0, 1)
#         pr_cum = pr_cum * scale_factor

#         # Apply onset time to cumulative
#         pr_cum = self._apply_onset_time(pr_cum, xr_onset_time)

#         # Recompute rate from shifted cumulative and extend time axis
#         pr = pr_cum.diff('time') / self.timestep
#         pr = pr.reindex({'time': extended_time}).fillna(0)
#         pr.name = 'pr'
#         pr.attrs = {'standard_name': 'precipitation rate', 'units': 'mm/hr'}

#         # Extend cumulative time axis
#         pr_cum = pr_cum.reindex({'time': extended_time}, method='pad')
#         pr_cum.name = 'pr_cum'
#         pr_cum.attrs = {
#             'standard_name': 'cumulative precipitation', 'units': 'mm'}

#         return pr, pr_cum

#     def _compute_dimensionless_storm(self,
#                                      xr_duration: xr.DataArray,
#                                      xr_rainfall: xr.DataArray,
#                                      xr_onset_time: xr.DataArray,
#                                      base_time: np.ndarray,
#                                      extended_time: np.ndarray,
#                                      interp_kwargs: dict
#                                      ) -> tuple[xr.DataArray, xr.DataArray]:
#         """
#         Compute storm using dimensionless hyetograph method.
#         Returns pr and pr_cum with onset time already applied.
#         """
#         shyeto = self._dimensionless_shyeto(base_time, xr_duration,
#                                             interp_kwargs)
#         return self._build_outputs(shyeto, xr_rainfall, xr_onset_time,
#                                    base_time, extended_time)

#     def compute(self,
#                 duration: int | float | ArrayLike,
#                 rainfall: ArrayLike = None,
#                 idf_curve: ArrayLike = None,
#                 onset_time: int | float | ArrayLike = 0,
#                 tail_duration: float = 0,
#                 idf_scaling: bool = False,
#                 interp_kwargs: dict = {'method': 'linear'},
#                 idf_kwargs: dict = {}
#                 ) -> Type['RainStorm']:
#         """
#         Trigger computation of design storm for a given storm duration
#         and total precipitation.

#         Args:
#             duration (float or array_like): Storm duration in hours.
#                 Can be scalar or broadcastable to rainfall/idf_curve (e.g.,
#                 spatial grid with same dims).
#             rainfall (array_like or float, optional): Total precipitation
#                 amount in mm. If provided as a scalar/array and
#                 `idf_scaling=False` (default), the same total rainfall is
#                 used for all target durations. If `idf_scaling=True`, IDF
#                 scaling is applied via duration coefficients so the total
#                 rainfall changes with the target durations. Cannot be used
#                 together with `idf_curve`.
#             onset_time (float or array_like, optional): Time delay in hours for
#                 storm onset. If scalar, shifts all precipitation uniformly.
#                 If array_like, allows spatially variable timing. Default is 0.
#             tail_duration (float, optional): Additional time in hours to extend
#                 the time axis after the storm ends. Used to pad with zeros after
#                 precipitation. Default is 0.
#             idf_scaling (bool, optional): Whether to apply IDF scaling when
#                 using `rainfall` parameter. Defaults to False. Ignored if
#                 `idf_curve` is provided.
#             idf_curve (array_like or xarray.DataArray, optional):
#                 Intensity-duration curve with a 'duration' coordinate
#                 representing intensity (mm/hr) for each duration (hr).
#                 When provided, the curve is internally converted to a
#                 rainfall-duration curve and evaluated at target durations.
#                 `rainfall` and `idf_scaling` are ignored. Can have additional
#                 dims (e.g., 'return_period', spatial dims). Both idf_curve
#                 and rdf_curve are stored as class attributes.
#             interp_kwargs (dict): Extra arguments for interpolation.
#                 Defaults to {'method': 'linear'}.
#             idf_kwargs (dict): Additional arguments passed to duration_coef
#                 when idf_scaling=True. Defaults to {}.

#         Returns:
#             Updated class instance with computed storm.
#         """
#         # Validate input parameters
#         if (rainfall is not None) == (idf_curve is not None):
#             raise ValueError(
#                 "Must specify exactly one of 'rainfall' or 'idf_curve'")

#         inputs = self._build_inputs(duration, rainfall, idf_curve, onset_time)
#         (xr_duration, xr_rainfall, xr_idf_curve, xr_onset_time) = inputs

#         # Build and store RDF/IDF curves
#         if xr_idf_curve is not None:
#             # Path 1: User provided IDF curve
#             self.idf_curve = self._validate_idf_curve(xr_idf_curve)
#             self.rdf_curve = self._compute_rdf_from_idf(self.idf_curve)
#             xr_rainfall = self._evaluate_rdf(self.rdf_curve, xr_duration,
#                                              interp_kwargs)
#         else:
#             # Path 2: User provided rainfall - build synthetic curves
#             # Force IDF scaling for alternating_block method
#             if self.kind == 'alternating_block':
#                 idf_scaling = True

#             # Build RDF curve from rainfall (with or without IDF scaling)
#             # The curve building handles scaling internally
#             self.rdf_curve = self._build_rdf_curve_from_rainfall(
#                 xr_rainfall, idf_scaling=idf_scaling, idf_kwargs=idf_kwargs
#             )
#             self.idf_curve = self._compute_idf_from_rdf(self.rdf_curve)

#             # Evaluate the RDF curve at target duration for non-alternating methods
#             if idf_scaling:
#                 xr_rainfall = self._evaluate_rdf(self.rdf_curve, xr_duration,
#                                                  interp_kwargs)

#         # # Build time axes
#         # base_time, extended_time = self._build_time_axes(total_duration,
#         #                                                  tail_duration)

#         # # Generate hyetograph using appropriate method
#         # if self.kind == 'alternating_block':
#         #     pr, pr_cum = self._compute_alternating_block_storm(
#         #         xr_duration, xr_onset_time, base_time, extended_time,
#         #         interp_kwargs
#         #     )
#         # else:
#         #     pr, pr_cum = self._compute_dimensionless_storm(
#         #         xr_duration, xr_rainfall, xr_onset_time,
#         #         base_time, extended_time, interp_kwargs
#         #     )

#         # # Update class attributes
#         # self.duration, self.rainfall = duration, rainfall
#         # self.pr, self.pr_cum, self.time = pr, pr_cum, pr.time.values
#         # return self
