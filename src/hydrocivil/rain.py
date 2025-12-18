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


def alternating_block_sort(arr: ArrayLike, axis=0):
    """
    Sorts an array in an alternating block pattern along a specified axis.

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
    return rearranged

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
        self.idf_curve = None
        self.rdf_curve = None
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
                      onset_time: ArrayLike = 0,
                      idf_curve: ArrayLike = None
                      ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray,
                                 xr.DataArray, float]:
        """
        Broadcast duration/rainfall/onset_time/idf_curve, validate values,
        return max total duration.
        """
        xr_duration = obj_to_xarray(duration).squeeze()
        xr_rainfall = obj_to_xarray(
            rainfall).squeeze() if rainfall is not None else None
        xr_onset_time = obj_to_xarray(onset_time).squeeze()
        xr_idf_curve = obj_to_xarray(
            idf_curve).squeeze() if idf_curve is not None else None

        # Rename generic dimension names to avoid broadcasting conflicts
        for da, prefix in [(xr_duration, 'duration'),
                           (xr_rainfall, 'rainfall'),
                           (xr_onset_time, 'onset_time'),
                           (xr_idf_curve, 'idf_curve')]:
            if da is None:
                continue
            if 'dim_' in str(da.dims):
                new_dims = {dim: f'{prefix}_{i}'
                            for i, dim in enumerate(da.dims)
                            if dim.startswith('dim_')
                            }
                da = da.rename(new_dims)
                if prefix == 'duration':
                    xr_duration = da
                elif prefix == 'rainfall':
                    xr_rainfall = da
                elif prefix == 'onset_time':
                    xr_onset_time = da
                else:
                    xr_idf_curve = da

        # Broadcast to same shape, avoiding idf_curve's 'duration' dim
        if xr_idf_curve is not None and 'duration' in xr_idf_curve.dims:
            # Broadcast xr_duration against idf_curve's other dims
            # (excluding curve)
            idf_other_dims = xr_idf_curve
            # Create a view with other dims only by taking a single
            # slice along the curve dimension; this preserves other dims
            # for broadcast
            idf_other_dims = idf_other_dims.isel(
                duration=0, drop=True)
            xr_duration, _ = xr.broadcast(xr_duration, idf_other_dims)
            xr_duration, xr_onset_time = xr.broadcast(xr_duration,
                                                      xr_onset_time)
        elif xr_rainfall is not None:
            xr_duration, xr_rainfall = xr.broadcast(xr_duration,
                                                    xr_rainfall)
            xr_duration, xr_onset_time = xr.broadcast(xr_duration,
                                                      xr_onset_time)
        else:
            xr_duration, xr_onset_time = xr.broadcast(xr_duration,
                                                      xr_onset_time)

        # Validate inputs
        if (xr_duration < 0).any():
            raise ValueError("duration must be non-negative")
        if xr_rainfall is not None and (xr_rainfall < 0).any():
            raise ValueError("rainfall must be non-negative")
        if xr_idf_curve is not None and (xr_idf_curve < 0).any():
            raise ValueError("idf_curve must be non-negative")
        if (xr_onset_time < 0).any():
            raise ValueError("onset_time must be non-negative")

        # Calculate total duration (max duration + max shift)
        duration_max = float(xr_duration.max())
        shift_max = float(xr_onset_time.max())
        total_duration = duration_max + shift_max

        return (xr_duration, xr_rainfall, xr_onset_time,
                xr_idf_curve, total_duration)

    def _apply_idf_scaling(self, xr_duration: xr.DataArray,
                           xr_rainfall: xr.DataArray,
                           idf_kwargs: dict) -> xr.DataArray:
        """
            Scale rainfall to target durations using duration_coef
            element-wise.
        """
        coef = duration_coef(xr_duration, **idf_kwargs)
        return xr_rainfall * coef

    def _evaluate_idf_and_build_rdf(self,
                                    xr_idf_curve: xr.DataArray,
                                    xr_duration: xr.DataArray,
                                    interp_kwargs: dict
                                    ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Evaluate an intensity-duration curve and convert to rainfall.

        Takes IDF (intensity in mm/hr) and converts to RDF (rainfall in mm)
        by multiplying by duration: rainfall = intensity * duration.
        Returns both the stored IDF curve and computed RDF curve.

        Args:
            xr_idf_curve: Intensity-duration curve with 'duration' coordinate
            xr_duration: Target durations for evaluation
            interp_kwargs: Interpolation keyword arguments

        Returns:
            Tuple of (idf_curve_stored, rdf_curve_computed)
        """
        if 'duration' not in xr_idf_curve.dims:
            raise ValueError("idf_curve must have a 'duration' dimension to "
                             "be evaluated as a curve")

        # Ensure monotonic increasing duration coordinate
        dur_series = xr_idf_curve['duration'].to_series()
        if not dur_series.is_monotonic_increasing:
            xr_idf_curve = xr_idf_curve.sortby('duration')

        # Validate that target durations are within curve bounds
        dmin = float(xr_idf_curve['duration'].min())
        dmax = float(xr_idf_curve['duration'].max())

        # Check if any requested duration is outside bounds
        if (xr_duration < dmin).any() or (xr_duration > dmax).any():
            raise ValueError(
                f"Cannot compute storm for durations outside IDF curve bounds. "
                f"IDF curve supports durations: [{dmin}, {dmax}] hr. "
                f"Requested durations: min={float(xr_duration.min())}, "
                f"max={float(xr_duration.max())} hr."
            )

        target_duration = xr_duration

        # Evaluate IDF at target durations
        xr_intensity = xr_idf_curve.interp(duration=target_duration,
                                           **interp_kwargs)
        # Convert intensity to rainfall: rain = intensity * duration
        xr_rdf = xr_intensity * target_duration
        # Ensure rainfall is 0 at duration 0 (physically correct)
        xr_rdf = xr_rdf.where(target_duration > 0, 0)

        return xr_idf_curve, xr_rdf

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

    def _apply_onset_time(self, pr_cum: xr.DataArray,
                          xr_onset_time: xr.DataArray
                          ) -> xr.DataArray:
        """
        Apply onset time to cumulative precipitation using 1D routine
        generalized with xr.apply_ufunc.

        Args:
            pr_cum: Cumulative precipitation with 'time' dimension
            xr_onset_time: Onset time values (scalar or spatial array)

        Returns:
            Onset-time-shifted cumulative precipitation
        """
        # If all shifts are zero, skip processing
        if (xr_onset_time == 0).all():
            return pr_cum

        # Apply 1D shift function across spatial dimensions
        orig_dims = pr_cum.dims
        pr_cum_shifted = xr.apply_ufunc(
            self._shift_timeseries_1d,
            pr_cum,
            xr_onset_time,
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
        shyeto = source_cum.interp(source_time=norm_time, **interp_kwargs
                                   )
        # Normalize final value to 1.0 to ensure pr_cum matches rainfall
        shyeto = shyeto / shyeto.isel(time=-1, drop=False)
        shyeto = shyeto.assign_coords(time=base_time)
        shyeto['time'].attrs = {'standard_name': 'time', 'units': 'hr'}
        return shyeto

    def _build_outputs(self, shyeto: xr.DataArray,
                       xr_rainfall: xr.DataArray,
                       xr_onset_time: xr.DataArray,
                       base_time: np.ndarray,
                       extended_time: np.ndarray
                       ) -> tuple[xr.DataArray, xr.DataArray]:
        """
        Create cumulative and rate precipitation fields, apply time
        shift, pad zeros beyond each duration.
        """
        pr_cum = shyeto * xr_rainfall
        # Ensure 'time' is the leading dimension regardless of how
        # inputs broadcast
        other_dims = [d for d in pr_cum.dims if d != 'time']
        pr_cum = pr_cum.transpose(*(['time'] + other_dims))

        # Apply onset time
        pr_cum = self._apply_onset_time(pr_cum, xr_onset_time)

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
                rainfall: ArrayLike = None,
                onset_time: int | float | ArrayLike = 0,
                tail_duration: float = 0,
                idf_scaling: bool = False,
                idf_curve: ArrayLike = None,
                interp_kwargs: dict = {'method': 'linear'},
                **idf_kwargs
                ) -> Type['RainStorm']:
        """
        Trigger computation of design storm for a given storm duration
        and total precipitation.

        Args:
            duration (float or array_like): Storm duration in hours.
                Can be scalar or broadcastable to rainfall/idf_curve (e.g.,
                spatial grid with same dims).
            rainfall (array_like or float, optional): Total precipitation
                amount in mm. If provided as a scalar/array and
                `idf_scaling=False` (default), the same total rainfall is
                used for all target durations. If `idf_scaling=True`, IDF
                scaling is applied via duration coefficients so the total
                rainfall changes with the target durations. Cannot be used
                together with `idf_curve`.
            onset_time (float or array_like, optional): Time delay in hours for
                storm onset. If scalar, shifts all precipitation uniformly.
                If array_like, allows spatially variable timing. Default is 0.
            tail_duration (float, optional): Additional time in hours to extend
                the time axis after the storm ends. Used to pad with zeros after
                precipitation. Default is 0.
            idf_scaling (bool, optional): Whether to apply IDF scaling when
                using `rainfall` parameter. Defaults to False. Ignored if
                `idf_curve` is provided.
            idf_curve (array_like or xarray.DataArray, optional):
                Intensity-duration curve with a 'duration' coordinate
                representing intensity (mm/hr) for each duration (hr).
                When provided, the curve is internally converted to a
                rainfall-duration curve and evaluated at target durations.
                `rainfall` and `idf_scaling` are ignored. Can have additional
                dims (e.g., 'return_period', spatial dims). Both idf_curve
                and rdf_curve are stored as class attributes.
            interp_kwargs (dict): extra arguments for interpolation
            **idf_kwargs: Additional arguments passed to duration_coef
                or other storm-specific parameters.

        Returns:
            Updated class instance with computed storm.
        """
        # Validate that only one of rainfall or idf_curve is provided
        if rainfall is not None and idf_curve is not None:
            raise ValueError(
                "Cannot specify both 'rainfall' and 'idf_curve'. "
                "Use 'rainfall' for constant amounts or 'idf_curve' "
                "for intensity-duration curves."
            )
        if rainfall is None and idf_curve is None:
            raise ValueError(
                "Must specify either 'rainfall' or 'idf_curve'."
            )

        (xr_duration, xr_rainfall, xr_onset_time, xr_idf_curve,
         total_duration) = self._build_inputs(
            duration, rainfall, onset_time, idf_curve
        )

        # Decide path: evaluate IDF curve vs duration or use rainfall
        if xr_idf_curve is not None:
            # Convert IDF to RDF and return both
            self.idf_curve, xr_rainfall = (
                self._evaluate_idf_and_build_rdf(
                    xr_idf_curve=xr_idf_curve,
                    xr_duration=xr_duration,
                    interp_kwargs=interp_kwargs
                )
            )
        else:
            # Using rainfall parameter - build RDF curve
            durations_rdf = np.arange(0, 72 + self.timestep,
                                      self.timestep)
            # Extract scalar value if rainfall is 0-D
            rainfall_base = (float(xr_rainfall) if xr_rainfall.ndim == 0
                             else xr_rainfall)

            if idf_scaling:
                # Build RDF curve with duration coefficients
                coef_values = duration_coef(durations_rdf, **idf_kwargs)
                if isinstance(rainfall_base, (float, int)):
                    rainfall_rdf = rainfall_base * coef_values
                    xr_rdf_curve = xr.DataArray(
                        rainfall_rdf,
                        dims=['duration'],
                        coords={'duration': durations_rdf}
                    )
                else:
                    # rainfall_base is array-like with spatial dims
                    coef_da = xr.DataArray(
                        coef_values,
                        dims=['duration'],
                        coords={'duration': durations_rdf}
                    )
                    xr_rdf_curve = rainfall_base * coef_da
                xr_rainfall = self._apply_idf_scaling(xr_duration,
                                                      xr_rainfall,
                                                      idf_kwargs)
            else:
                # Build RDF curve with constant rainfall
                if isinstance(rainfall_base, (float, int)):
                    rainfall_rdf = rainfall_base * np.ones_like(
                        durations_rdf
                    )
                    xr_rdf_curve = xr.DataArray(
                        rainfall_rdf,
                        dims=['duration'],
                        coords={'duration': durations_rdf}
                    )
                else:
                    # rainfall_base is array-like with spatial dims
                    ones_da = xr.DataArray(
                        np.ones_like(durations_rdf),
                        dims=['duration'],
                        coords={'duration': durations_rdf}
                    )
                    xr_rdf_curve = rainfall_base * ones_da
            # Ensure rainfall is 0 at duration 0 (physically correct)
            xr_rdf_curve.loc[{'duration': 0}] = 0
            xr_rdf_curve.name = 'rainfall'
            xr_rdf_curve.attrs = {
                'standard_name': 'total_rainfall_vs_duration',
                'units': 'mm'
            }
            self.rdf_curve = xr_rdf_curve

            # Also compute and store corresponding IDF curve
            # IDF = RDF / duration (intensity = rainfall / duration)
            xr_idf_curve = xr_rdf_curve / durations_rdf
            # Ensure intensity is inf at duration 0 (rainfall/0 = inf)
            xr_idf_curve = xr_idf_curve.where(
                durations_rdf > 0, np.inf)
            xr_idf_curve.name = 'intensity'
            xr_idf_curve.attrs = {
                'standard_name': 'intensity_vs_duration',
                'units': 'mm/hr'
            }
            self.idf_curve = xr_idf_curve

        base_time, extended_time = self._build_time_axes(total_duration,
                                                         tail_duration)

        shyeto = self._dimensionless_shyeto(base_time, xr_duration,
                                            interp_kwargs)

        pr, pr_cum = self._build_outputs(shyeto, xr_rainfall,
                                         xr_onset_time,
                                         base_time, extended_time)

        # Update class attributes
        self.duration = duration
        self.rainfall = rainfall
        self.pr = pr
        self.pr_cum = pr_cum
        self.time = pr.time.values
        return self
