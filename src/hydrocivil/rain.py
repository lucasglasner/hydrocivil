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
from .misc import obj_to_xarray, series2func

import scipy.stats as st

# -------------------- Intensity-duration-frequency curves ------------------- #


def grunsky_coef(storm_duration: int | float,
                 ref_duration: int | float = 24,
                 expon: float = 0.5) -> float:
    """
    This function computes the duration coefficient given by a Grunsky-like
    Formula. Those formulas state that the duration coefficient is a power
    law of the storm duration t:

        Cd (t) = (t / ref) ^ b

    Where "ref" represents the reference duration, typically 24 hours, "t" is
    the storm duration of interest, and "b" is an empirical parameter.
    The traditional Grunsky formula assumes b = 0.5 , which is generally valid
    for cyclonic precipitation on flat terrain. However, for convective
    rainfall or rainfall on complex terrain, a different value of b may apply.

    References:
        ???

    Args:
        storm_duration (array_like): storm duration in (hours)
        expon (float): Exponent of the power law. Defaults to 0.5 (Grunsky).
        ref_duration (array_like): Reference rain duration (hours).
            Defaults to 24 hr

    Returns:
        CD (array_like): Duration coefficient in (dimensionless)
    """
    CD = (storm_duration/ref_duration)**expon
    return CD


def bell_coef(storm_duration: int | float, cd1: float = None,
              ref_duration: int | float = 24, expon: float = 0.5) -> float:
    """
    This function computes the duration coefficient
    given by the Bell Formula.

    References:
        Bell, F.C. (1969) Generalized pr-Duration-Frequency
        Relationships. Journal of Hydraulic Division, ASCE, 95, 311-327.

    Args:
        storm_duration (array_like): duration in (hours)
        cd1 (float, optional): Duration coefficient for 1 hour duration.
            Defaults to None, which will use the Grunsky coefficient for 1 hr.
        ref_duration (array_like): Reference rain duration (hours).
            Defaults to 24 hr.
        expon (float): Exponent of the Grunsky power law. Defaults to 0.5.

    Returns:
        CD (array_like): Duration coefficient in (dimensionless)
    """
    a = (0.54*((storm_duration*60)**0.25)-0.5)
    if cd1 is None:
        b = grunsky_coef(1, ref_duration, expon=expon)
    else:
        b = cd1
    CD = a*b
    return CD


def duration_coef(storm_duration: int | float, ref_pgauge: str = 'Grunsky',
                  ref_duration: int | float = 24, expon: float = 0.5,
                  **kwargs) -> float:
    """
    Compute duration coefficient to convert precipitation from a reference
    duration (typically 24 hr) to a target duration. Used to estimate
    rainfall for different durations based on empirical formulas.

    - For durations >= 1 hr, Grunsky's formula is used (power law).
    - For durations < 1 hr, Bell's formula is used.
    - If a reference rain gauge is specified (other than 'Grunsky'),
      coefficients are interpolated from predefined tables, with Bell's
      formula for durations < 1 hr.

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
        ref_duration (int or float, optional): Reference duration [hr].
            Default is 24.
        expon (float): Exponent of the Grunsky power law. Defaults to 0.5.


    Returns:
        np.ndarray: Duration coefficient(s) (dimensionless).
    """
    durations = np.asarray(storm_duration, dtype=float)
    scalar_input = durations.ndim == 0
    durations = np.atleast_1d(durations)

    bell_mask = durations < 1
    if ref_pgauge == 'Grunsky':
        coefs = grunsky_coef(durations, ref_duration=ref_duration,
                             expon=expon)
        if bell_mask.any():
            cd1 = grunsky_coef(1.0, ref_duration=ref_duration, expon=expon)
            coefs = coefs.copy()
            coefs[bell_mask] = bell_coef(durations[bell_mask], cd1=cd1,
                                         ref_duration=ref_duration)
    else:
        if ref_pgauge not in DURCOEFS_PGAUGES.columns:
            raise ValueError(f"Reference rain gauge '{ref_pgauge}' not found.")
        if ref_duration != 24:
            raise ValueError(
                "Reference duration must be 24 hr for tabulated "
                "duration coefficients.")
        func = series2func(DURCOEFS_PGAUGES[ref_pgauge], **kwargs)
        coefs = func(durations)
        if bell_mask.any():
            cd1 = DURCOEFS_PGAUGES[ref_pgauge].loc[1]
            coefs = coefs.copy()
            coefs[bell_mask] = bell_coef(durations[bell_mask], cd1=cd1,
                                         ref_duration=ref_duration)

    coefs = np.clip(coefs, 0, None)
    return coefs if not scalar_input else coefs[0]

# ------------------------------- Design Storms ------------------------------ #


class RainStorm:
    """
    The class can be used to build rainstorms that follow any of scipy
    theoretical distributions (e.g 'norm', 'skewnorm', 'gamma', etc) or
    the empirical rain distributions of the SCS type I, IA, II, III and the
    Chilean synthetic hyetographs of (Espildora and Echavarría 1979),
    (Benitez and Verni 1985) and (Varas 1985).
    """
    PREDEFINED_STORMS = SHYETO_DATA.columns

    def __init__(self, kind: str, timestep: float, **kwargs: Any) -> None:
        """
        Synthetic RainStorm builder

        Args:
            kind (str): Type of storm model to use.
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
                   G4p75_Varas1985, G4p90_Varas1985
                - SciPy distribution (e.g., 'norm', 'gamma')
            timestep (float): Storm timestep or resolution in hours.
            **kwargs: Additional parameters depending on the storm type.
                - For predefined storms: No extra parameters needed.
                - For SciPy distributions: `loc`, `scale`, and shape params.

        Examples:
            RainStorm('SCS_I24', timestep=0.5)
            RainStorm('G2_Benitez1985', timestep=0.5)
            RainStorm('G3_Espildora1979', timestep=0.5)
            RainStorm('G4p10_Varas1985', timestep=0.5)
            RainStorm('norm', timestep=0.5, loc=0.5, scale=0.2)
            RainStorm('gamma', timestep=0.5, loc=0, scale=0.15, a=2)
        """
        if timestep <= 0:
            raise ValueError(f"timestep must be positive, got {timestep}")

        self.kind = kind
        self.timestep = timestep
        self.duration = None
        self.rainfall = None
        self.pr = None
        self.pr_cum = None
        self.time = None

        if kind in self.PREDEFINED_STORMS:
            self.pr_dimless = self._predefined_hyetograph(kind)
        elif hasattr(st, kind):
            self.pr_dimless = self._scipy_hyetograph(kind, **kwargs)
        else:
            raise ValueError(f"Unknown storm type: {kind}")

        # Cache cumulative hyetograph
        self._pr_dimless_cum = self.pr_dimless.cumsum()

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

        Returns:
            (pandas.Series): Synthetic Hyetograph 1D Table
        """
        return SHYETO_DATA[kind]

    def _scipy_hyetograph(self, kind: str, loc: float = 0.5, scale: float = 0.1,
                          flip: bool = False, n: int = 1000, **kwargs: Any
                          ) -> pd.Series:
        """Synthetic hyetograph generator function for any of scipy
        distributions. The synthetic hyetograph will be built with the given
        loc, scale and scipy default parameters.

        Args:
            loc (float, optional): Location parameter for distribution type
                hyetographs. Defaults to 0.5.
            scale (float, optional): Scale parameter for distribution type
                hyetographs. Defaults to 0.1.
            flip (bool): Whether to flip the distribution along the x-axis
                or not. Defaults to False.
            n (int, optional): Number of records in the dimensionless storm
            **kwargs are given to scipy.rv_continuous.pdf

        Returns:
            (pandas.Series): Synthetic Hyetograph 1D Table
        """
        time_dimless = np.linspace(0, 1, n)
        distr = getattr(st, kind)
        shyeto = distr.pdf(time_dimless, loc=loc, scale=scale, **kwargs)
        shyeto /= np.sum(shyeto)  # Normalize to sum 1
        if flip:
            shyeto = shyeto[::-1]
        return pd.Series(shyeto, index=time_dimless)

    def copy(self) -> Type['RainStorm']:
        """
        Create a deep copy of the class itself
        """
        return pycopy.deepcopy(self)

    def _build_inputs(self, duration: ArrayLike, rainfall: ArrayLike,
                      time_shift: ArrayLike = 0
                      ) -> tuple[xr.DataArray, xr.DataArray,
                                 xr.DataArray, float, dict]:
        """
        Broadcast duration/rainfall/time_shift, validate values,
        return max total duration and dims.
        """
        xr_duration = obj_to_xarray(duration).squeeze()
        xr_rainfall = obj_to_xarray(rainfall).squeeze()
        xr_time_shift = obj_to_xarray(time_shift).squeeze()

        # Broadcast all to same shape
        xr_duration, xr_rainfall = xr.broadcast(xr_duration,
                                                xr_rainfall)
        xr_duration, xr_time_shift = xr.broadcast(xr_duration,
                                                  xr_time_shift)

        # Validate inputs
        if (xr_duration < 0).any():
            raise ValueError("duration must be non-negative")
        if (xr_rainfall < 0).any():
            raise ValueError("rainfall must be non-negative")
        if (xr_time_shift < 0).any():
            raise ValueError("time_shift must be non-negative")

        # Calculate total duration (max duration + max shift)
        duration_max = float(xr_duration.max())
        shift_max = float(xr_time_shift.max())
        total_duration = duration_max + shift_max

        dims = {dim: xr_rainfall[dim].shape[0]
                for dim in xr_rainfall.dims}
        return (xr_duration, xr_rainfall, xr_time_shift,
                total_duration, dims)

    def _apply_idf_scaling(self, xr_duration: xr.DataArray,
                           xr_rainfall: xr.DataArray,
                           idf_kwargs: dict) -> xr.DataArray:
        """
            Scale rainfall to target durations using duration_coef
            element-wise.
        """
        coef = duration_coef(xr_duration, **idf_kwargs)
        return xr_rainfall * coef

    def _build_time_axes(self, total_duration: float,
                         tail_duration: float
                         ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build base and extended time arrays covering total duration
        plus optional tail padding.

        Args:
            total_duration: Maximum storm duration + max time shift (hr)
            tail_duration: Additional time to pad after storm ends (hr)

        Returns:
            base_time: Time array for storm duration
            extended_time: Time array including tail padding
        """
        base_time = np.arange(0, total_duration + self.timestep*1e-3,
                              self.timestep)
        extended_time = np.arange(
            0, total_duration + tail_duration + self.timestep*1e-3,
            self.timestep
        )
        return base_time, extended_time

    def _shift_timeseries_1d(self, cum_series: np.ndarray,
                             shift_hours: float) -> np.ndarray:
        """
        Shift a 1D cumulative time series forward in time by
        prepending zeros.

        Args:
            cum_series: 1D array of cumulative precipitation values
            shift_hours: Hours to shift forward (>= 0)

        Returns:
            Shifted cumulative series (same length as input)
        """
        if shift_hours <= 0:
            return cum_series

        # Calculate number of time steps to shift
        shift_steps = int(np.round(shift_hours / self.timestep))
        n_total = len(cum_series)

        # Prepend zeros and trim to original length
        shifted = np.concatenate([
            np.zeros(shift_steps),
            cum_series
        ])
        return shifted[:n_total]

    def _apply_time_shift(self, pr_cum: xr.DataArray,
                          xr_time_shift: xr.DataArray
                          ) -> xr.DataArray:
        """
        Apply time shift to cumulative precipitation using 1D routine
        generalized with xr.apply_ufunc.

        Args:
            pr_cum: Cumulative precipitation with 'time' dimension
            xr_time_shift: Time shift values (scalar or spatial array)

        Returns:
            Time-shifted cumulative precipitation
        """
        # If all shifts are zero, skip processing
        if (xr_time_shift == 0).all():
            return pr_cum

        # Apply 1D shift function across spatial dimensions
        orig_dims = pr_cum.dims
        pr_cum_shifted = xr.apply_ufunc(
            self._shift_timeseries_1d,
            pr_cum,
            xr_time_shift,
            input_core_dims=[['time'], []],
            output_core_dims=[['time']],
            vectorize=True,
        )

        # Restore original dimension order
        pr_cum_shifted = pr_cum_shifted.transpose(*orig_dims)
        return pr_cum_shifted

    def _dimensionless_shyeto(self, base_time: np.ndarray,
                              xr_duration: xr.DataArray,
                              interp_kwargs: dict) -> xr.DataArray:
        """
            Interpolate cached dimensionless cumulative hyetograph over
            normalized time per cell.
        """
        source_cum = xr.DataArray(
            self._pr_dimless_cum.values,
            dims=['source_time'],
            coords={'source_time': self.pr_dimless.index.values}
        )
        norm_time = (xr.DataArray(base_time, dims=['time']) /
                     xr_duration).clip(0, 1)
        shyeto = source_cum.interp(
            source_time=norm_time, **interp_kwargs
        )
        # Normalize final value to exactly 1.0 to ensure pr_cum matches
        # rainfall
        shyeto = shyeto / shyeto.isel(time=-1, drop=False)
        shyeto = shyeto.assign_coords(time=base_time)
        shyeto['time'].attrs = {'standard_name': 'time', 'units': 'hr'}
        return shyeto

    def _build_outputs(self, shyeto: xr.DataArray,
                       xr_rainfall: xr.DataArray,
                       xr_time_shift: xr.DataArray,
                       dims: dict, base_time: np.ndarray,
                       extended_time: np.ndarray
                       ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Create cumulative and rate precipitation fields, apply time
        shift, pad zeros beyond each duration.
        """
        pr_cum = shyeto * xr_rainfall
        pr_cum = pr_cum.transpose(*(['time'] + list(dims.keys())))

        # Apply time shift
        pr_cum = self._apply_time_shift(pr_cum, xr_time_shift)

        pr = (pr_cum.diff('time').reindex({'time': base_time}) /
              self.timestep)
        pr = pr.reindex({'time': extended_time}).fillna(0)

        pr.name = 'pr'
        pr.attrs = {
            'standard_name': 'precipitation rate',
            'units': 'mm/hr'
        }

        pr_cum = pr_cum.reindex({'time': extended_time}, method='pad')
        pr_cum.name = 'pr_cum'
        pr_cum.attrs = {
            'standard_name': 'cumulative precipitation',
            'units': 'mm'
        }
        return pr, pr_cum

    def compute(self, duration: int | float | ArrayLike,
                rainfall: ArrayLike,
                time_shift: int | float | ArrayLike = 0,
                tail_duration: float = 0,
                interp_kwargs: dict = {'method': 'linear'},
                **idf_kwargs
                ) -> Type['RainStorm']:
        """
        Trigger computation of design storm for a given storm duration
        and total precipitation.

        Args:
            duration (float or array_like): Storm duration in hours.
                Can be scalar or broadcastable to rainfall (e.g.,
                spatial grid with same dims). Variable durations are
                supported; the time axis will span the maximum duration
                and shorter events are zero-padded after their own
                duration.
            rainfall (array_like or float): Total precipitation in mm.
            time_shift (float or array_like, optional): Time delay in
                hours to shift storm onset. If scalar, shifts all
                precipitation uniformly. If array_like, allows
                spatially variable timing (e.g., frontal systems).
                Default is 0.
            tail_duration (float, optional): Additional time in hours
                to extend the time axis after the storm ends. Used to
                pad with zeros after precipitation. Default is 0.
            interp_kwargs (dict): extra arguments for interpolation
            **idf_kwargs: Additional arguments passed to duration_coef

        Returns:
            Updated class instance with computed storm.
        """
        xr_duration, xr_rainfall, xr_time_shift, total_duration, dims = \
            self._build_inputs(duration, rainfall, time_shift)

        xr_rainfall = self._apply_idf_scaling(xr_duration,
                                              xr_rainfall,
                                              idf_kwargs)

        base_time, extended_time = self._build_time_axes(total_duration,
                                                         tail_duration)

        shyeto = self._dimensionless_shyeto(base_time, xr_duration,
                                            interp_kwargs)

        pr, pr_cum = self._build_outputs(shyeto, xr_rainfall,
                                         xr_time_shift, dims,
                                         base_time, extended_time)

        # Update class attributes
        self.duration = duration
        self.rainfall = rainfall
        self.pr = pr
        self.pr_cum = pr_cum
        self.time = pr.time.values
        return self
