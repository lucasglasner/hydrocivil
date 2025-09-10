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
import warnings
import copy as pycopy
import xarray as xr

from typing import Any, Type
from numpy.typing import ArrayLike
from .abstractions import SCS_Abstractions, Horton_Abstractions
from .abstractions import Philip_Abstractions, GreenAmpt_Abstractions
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
        ref_pgauge (str, optional): Reference gauge name or 'Grunsky'.
        ref_duration (int or float, optional): Reference duration [hr].
            Default is 24.
        expon (float): Exponent of the Grunsky power law. Defaults to 0.5.


    Returns:
        np.ndarray: Duration coefficient(s) (dimensionless).
    """
    if np.isscalar(storm_duration):
        storm_duration = np.array([storm_duration])
    else:
        storm_duration = np.asarray(storm_duration)
    coefs = np.full(storm_duration.shape, np.nan)
    bell_mask = storm_duration < 1
    if ref_pgauge == 'Grunsky':
        coefs = grunsky_coef(storm_duration, ref_duration=ref_duration,
                             expon=expon)
        coefs[bell_mask] = bell_coef(storm_duration[bell_mask],
                                     ref_duration=ref_duration)
    else:
        if ref_pgauge not in DURCOEFS_PGAUGES.columns:
            raise ValueError(f"Reference rain gauge '{ref_pgauge}' not found.")
        if ref_duration != 24:
            raise ValueError(
                "Reference duration must be 24 hr for tabulated "
                "duration coefficients.")
        func = series2func(DURCOEFS_PGAUGES[ref_pgauge], **kwargs)
        CD1 = DURCOEFS_PGAUGES[ref_pgauge].loc[1]
        coefs = func(storm_duration)
        coefs[bell_mask] = bell_coef(storm_duration[bell_mask], cd1=CD1)
    coefs[coefs < 0] = 0
    return coefs

# ------------------------------- Design Storms ------------------------------ #


class RainStorm:
    """
    RainStorm class used to building temporal rainfall distributions. 
    The class can be used to build rainstorms that follow any of scipy
    theoretical distributions (e.g 'norm', 'skewnorm', 'gamma', etc) or 
    the empirical rain distributions of the SCS type I, IA, II, III and the 
    Chilean synthetic hyetographs of (Espildora and Echavarría 1979),
    (Benitez and Verni 1985) and (Varas 1985). 
    """
    PREDEFINED_STORMS = SHYETO_DATA.columns

    def __init__(self, kind: str, **kwargs: Any) -> None:
        """
        Synthetic RainStorm builder

        Args:
            kind (str): Type of storm model to use.
                - Predefined (e.g., 'SCS_I24', 'GX_Benitez1985_1')
                - SciPy distribution (e.g., 'norm', 'gamma')
            **kwargs: Additional parameters depending on the storm type.
                - For predefined storms: No extra parameters needed.
                - For SciPy distributions: `loc`, `scale`, and shape params.

        Examples:
            RainStorm('SCS_I24')
            RainStorm('G2_Benitez1985')
            RainStorm('G3_Espildora1979')
            RainStorm('G4p10_Varas1985')
            RainStorm('norm', loc=0.5, scale=0.2)
            RainStorm('gamma', loc=0, scale=0.15, a=2)
        """

        self.kind = kind
        self.timestep = None
        self.duration = None
        self.rainfall = None
        self.infiltration = None
        self.pr = None
        self.pr_eff = None
        self.infr = None

        if kind in self.PREDEFINED_STORMS:
            self.pr_dimless = self._predefined_hyetograph(kind)
        elif hasattr(st, kind):
            self.pr_dimless = self._scipy_hyetograph(kind, **kwargs)
        else:
            raise ValueError(f"Unknown storm type: {kind}")

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
            shyeto = pd.Series(shyeto[::-1], index=time_dimless)
        else:
            shyeto = pd.Series(shyeto, index=time_dimless)
        return pd.Series(shyeto, index=time_dimless)

    def __repr__(self) -> str:
        """
        What to show when invoking a RainStorm object
        Returns:
            str: Some metadata
        """
        text = f"RainStorm(kind='{self.kind}', timestep={self.timestep}, "
        text = text+f"duration={self.duration}, "
        text = text+f"infiltration='{self.infiltration}')"
        return text

    def copy(self) -> Type['RainStorm']:
        """
        Create a deep copy of the class itself
        """
        return pycopy.deepcopy(self)

    def compute(self, timestep: int | float, duration: int | float,
                rainfall: ArrayLike, n: int = 0,
                interp_kwargs: dict = {'method': 'linear'}
                ) -> Type['RainStorm']:
        """
        Trigger computation of design storm for a given timestep, storm 
        duration, and total precipitation.

        Args:
            timestep (float): Storm timestep or resolution in hours
            duration (float): Storm duration in hours
            rainfall (array_like or float): Total precipitation in mm. 
            n (int, optional): If n=0 the storm time length will be equal to 
                the user storm duration. If n>0 it will expand the time
                dimension with zeros n times the user storm duration.
            interp_kwargs (dict): extra arguments for the interpolation function

        Returns:
            Updated class
        """
        self.timestep = timestep
        self.duration = duration
        self.rainfall = rainfall

        xr_rainfall = obj_to_xarray(rainfall).squeeze()
        dims = {dim: xr_rainfall[dim].shape[0] for dim in xr_rainfall.dims}
        time1 = np.arange(0, duration+timestep*1e-3, timestep)
        time2 = np.arange(0, duration+n*duration+timestep*1e-3, timestep)

        # Build dimensionless storm (accumulates 1 mm)
        shyeto = obj_to_xarray(self.pr_dimless.cumsum(), dims=('time'),
                               coords={'time': self.pr_dimless.index})
        shyeto = shyeto.interp(coords={'time': np.linspace(0, 1, len(time1))},
                               **interp_kwargs)
        shyeto.coords['time'] = time1
        shyeto['time'].attrs = {'standard_name': 'time', 'units': 'hr'}

        # Build real cumulative precipitation for the given rainfall
        pr_cum = shyeto.expand_dims(dim=dims)*xr_rainfall
        pr_cum = pr_cum.transpose(*(['time']+list(dims.keys())))

        # Transform storm time series to a precipitation rate time series
        pr = pr_cum.diff('time').reindex({'time': time1})/timestep
        pr = pr.reindex({'time': time2}).fillna(0)

        # Metadata
        pr.name = 'pr'
        pr.attrs = {'standard_name': 'precipitation rate', 'units': 'mm/hr'}
        self.pr = pr
        self.pr_cum = pr_cum
        self.time = pr.time.values
        return self

    def _infiltrate_SCS(self, cn: float, r: float, **kwargs):
        """
        Compute infiltration rate using the SCS method.

        Args:
            cn (array_like or float): Curve Number
            **kwargs are passed to xr.apply_ufunc
            r (float): Initial abstraction ratio, default 0.2

        Returns:
            (array_like): Infiltration rate [mm/h]
        """
        # Compute losses
        pr_cum = self.pr.cumsum('time')*self.timestep  # Accumulate over time
        infr_cum = xr.apply_ufunc(SCS_Abstractions, pr_cum, cn, r,
                                  input_core_dims=[['time'], [], []],
                                  output_core_dims=[['time']],
                                  vectorize=True,
                                  **kwargs)
        # Compute infiltration rate
        infr = infr_cum.transpose(*self.pr.dims).diff('time')
        infr = infr.reindex({'time': self.time})/self.timestep
        infr[0] = infr_cum.isel(time=0)
        return infr

    def _infiltrate_Horton(self, f0: float, fc: float, k: float, **kwargs):
        """
        Compute infiltration rate using 3 parameter Horton's method.

        Args:
            pr (float|array): precipitation rate (mm/h)
            duration (float): Time duration of rainfall event (h)
            f0 (float): Dry or initial soil hydraulic conductivity (mm/h)
            fc (float): Saturated soil hydraulic conductivity (mm/h)
            k (float): Horton's method decay coefficient (1/h)
            **kwargs are passed to xr.apply_ufunc

        Returns:
            (array_like): Infiltration rate [mm/h]
        """
        # Compute losses
        infr = xr.apply_ufunc(Horton_Abstractions, self.pr, self.time,
                              f0, fc, k,
                              input_core_dims=[['time'], ['time'],
                                               [], [], []],
                              output_core_dims=[['time']],
                              vectorize=True, **kwargs)
        infr = infr.transpose(*self.pr.dims)
        return infr

    def _infiltrate_Philip(self, S: float, K: float, **kwargs):
        """
        Compute infiltration rate using 2 parameter Philip's method.

        Args:
            pr (float|array): precipitation rate (mm/h)
            duration (float): Time duration of rainfall event (h)
            S (float): Adsorption coefficient (mm / h ^ 0.5)
            K (float): Saturated soil hydraulic conductivity (mm/h)
            **kwargs are passed to xr.apply_ufunc

        Returns:
            (array_like): Infiltration rate [mm/h]
        """
        # Compute losses
        infr = xr.apply_ufunc(Philip_Abstractions, self.pr, self.time, S, K,
                              input_core_dims=[['time'], ['time'],
                                               [], []],
                              output_core_dims=[['time']],
                              vectorize=True, **kwargs)
        infr = infr.transpose(*self.pr.dims)
        return infr

    def _infiltrate_GreenAmpt(self, K: float, p: float, theta_s: float,
                              psi: float, h0: float = 10, **kwargs):
        """
        Compute infiltration rate using Green & Ampt soil model.

        Args:
            K (float): Saturated soil hydraulic conductivity (mm/h)
            p (float): Soil porosity (-)
            theta_s (float): Soil fractional moisture (-)
            psi (float): Soil suction (mm). Highly dependant of soil moisture.
            h0 (float): water depth above the soil column (mm).
            Default to 10 mm. 
            **kwargs are passed to xr.apply_ufunc

        Returns:
            (array_like): Infiltration rate [mm/h]
        """
        # Compute losses
        infr = xr.apply_ufunc(GreenAmpt_Abstractions, self.pr, self.time,
                              K, p, theta_s, psi, h0,
                              input_core_dims=[['time'], ['time'],
                                               [], [], [], [], []],
                              output_core_dims=[['time']],
                              vectorize=True, **kwargs)
        infr = infr.transpose(*self.pr.dims)
        return infr

    def infiltrate(self, method: str = 'SCS', **kwargs: Any
                   ) -> Type['RainStorm']:
        """
        Compute losses due to infiltration with different methods for the
        stored storm Hyetograph
        Args:
            method (str, optional): Infiltration routine. Defaults to 'SCS'.

        Returns:
            Updated class
        """
        if self.pr is None:
            text = "A storm must be computed before infiltration can "
            text += "be performed. Use the self.compute method."
            raise ValueError(text)

        self.infiltration = method
        if method == 'SCS':
            # Grab curve number from keyword arguments
            kwargs = kwargs.copy()
            cn = kwargs['cn']
            if 'r' in kwargs.keys():
                r = kwargs['r']
            else:
                r = 0.2
            kwargs.pop('r', None)
            kwargs.pop('cn', None)

            # Compute losses
            infr = self._infiltrate_SCS(cn=cn, r=r, **kwargs)

        elif method == 'Horton':
            # Grab parameters from keyword arguments
            f0 = kwargs['f0']
            fc = kwargs['fc']
            k = kwargs['k']
            kwargs = kwargs.copy()
            kwargs.pop('f0', None)
            kwargs.pop('fc', None)
            kwargs.pop('k', None)
            infr = self._infiltrate_Horton(f0=f0, fc=fc, k=k, **kwargs)

        elif method == 'Philip':
            # Grab parameters from keyword arguments
            S = kwargs['S']
            K = kwargs['K']
            kwargs = kwargs.copy()
            kwargs.pop('S', None)
            kwargs.pop('K', None)
            infr = self._infiltrate_Philip(S=S, K=K, **kwargs)

        elif method == 'GreenAmpt':
            # Grab parameters from keyword arguments
            K = kwargs['K']
            p = kwargs['p']
            theta_s = kwargs['theta_s']
            psi = kwargs['psi']
            if 'h0' in kwargs.keys():
                h0 = kwargs['h0']
            else:
                h0 = 10.
            kwargs = kwargs.copy()
            kwargs.pop('K', None)
            kwargs.pop('p', None)
            kwargs.pop('theta_s', None)
            kwargs.pop('psi', None)
            kwargs.pop('h0', None)
            infr = self._infiltrate_GreenAmpt(K=K, p=p, theta_s=theta_s,
                                              psi=psi, h0=h0, **kwargs)
        else:
            raise ValueError(f'{method} unknown infiltration method.')

        # Define effective precipitation and update variables
        pr_eff = self.pr-infr
        pr_eff = pr_eff.where(pr_eff >= 0).fillna(0)

        # Metadata
        infr.attrs = {'standard_name': 'infiltration rate', 'units': 'mm/hr'}
        pr_eff.attrs = {'standard_name': 'effective precipitation rate',
                        'units': 'mm/hr'}
        self.infr = infr
        self.pr_eff = pr_eff
        return self
