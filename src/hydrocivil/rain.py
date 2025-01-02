'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 1969-12-31 21:00:00
 Modified by: Lucas Glasner,
 Modified time: 2024-05-06 16:24:28
 Description:
 Dependencies:
'''

import os
import numpy as np
import pandas as pd
import warnings
import copy as pycopy
import xarray as xr

from typing import Union, Any, Type
from numpy.typing import ArrayLike
from .abstractions import SCS_Abstractions
from .global_vars import SHYETO_DATA
from .misc import obj_to_xarray

import scipy.stats as st

# ----------------------- duration coefficient routines ---------------------- #


def grunsky_coef(storm_duration: Union[int, float],
                 ref_duration: Union[int, float] = 24) -> float:
    """
    This function computes the duration coefficient
    given by the Grunsky Formula.

    References:
        ???

    Args:
        storm_duration (array_like): storm duration in (hours)
        ref_duration (array_like): Reference duration (hours).
            Defaults to 24 hr

    Returns:
        CD (array_like): Duration coefficient in (dimensionless)
    """
    CD = np.sqrt(storm_duration/ref_duration)
    return CD


def bell_coef(storm_duration: Union[int, float],
              ref_duration: Union[int, float] = 24) -> float:
    """
    This function computes the duration coefficient
    given by the Bell Formula.

    References:
        Bell, F.C. (1969) Generalized pr-Duration-Frequency
        Relationships. Journal of Hydraulic Division, ASCE, 95, 311-327.

    Args:
        storm_duration (array_like): duration in (hours)

    Returns:
        CD (array_like): Duration coefficient in (dimensionless)
    """
    a = (0.54*((storm_duration*60)**0.25)-0.5)
    b = grunsky_coef(1, ref_duration)
    CD = a*b
    return CD


def duration_coef(storm_duration: Union[int, float],
                  ref_duration: Union[int, float] = 24,
                  bell_threshold: Union[int, float] = 1,
                  duration_threshold: Union[int, float] = 10/60) -> float:
    """
    The duration coefficient is a parameter used to transform a known duration
    precipitation to a new duration rain. For example it can be used to
    estimate from daily rainfall (24 hours) the expected accumulation in
    6 hours. This function uses a merge of Grunsky and Bell Formulations of the
    duration coefficient. The idea is to use Bell's Formula only when the input
    duration is less than a user specified threshold. In addition, when the
    duration is less than the "duration_threshold" the duration is set to the
    "duration_threshold".

    References:
        Bell, F.C. (1969) Generalized Rainfall-Duration-Frequency
        Relationships. Journal of Hydraulic Division, ASCE, 95, 311-327.

        Grunsky (???)


    Args:
        storm_duration (array_like): Storm duration in hours
        bell_threshold (float, optional): Duration threshold for changing
            between Grunsky and Bell formulas. Defaults to 1 (hour).
        duration_threshold (float, optional): Minimum storm duration.
            Defaults to 10 minutes (1/6 hours).

    Returns:
        coef (array_like): Duration coefficients (dimensionless)
    """
    if np.isscalar(storm_duration):
        storm_duration = np.array([storm_duration])
    coefs = np.full(storm_duration.shape, np.nan)
    duration_mask = storm_duration < duration_threshold
    bell_mask = storm_duration < bell_threshold
    if duration_mask.sum() != 0:
        threshold = f'{duration_threshold*60:.1f}'
        text = f'A storm duration is less than {threshold} min threshold,'
        text = text+f' setting to {threshold} min.'
        warnings.warn(text)
        storm_duration[duration_mask] = duration_threshold
    coefs[bell_mask] = bell_coef(storm_duration[bell_mask],
                                 ref_duration=ref_duration)
    coefs[~bell_mask] = grunsky_coef(storm_duration[~bell_mask],
                                     ref_duration=ref_duration)
    return coefs


# ------------------------------- Design Storms ------------------------------ #


class RainStorm(object):
    """
    RainStorm class used to building temporal rainfall distributions. 
    The class can be used to build rainstorms that follow any of scipy
    theoretical distributions (e.g 'norm', 'skewnorm', 'gamma', etc) or 
    the empirical rain distributions of the SCS type I, IA, II, III and the 
    Chilean synthetic hyetographs of (Espildora and Echavarría 1979),
    (Benitez and Verni 1985) and (Varas 1985). 

    Examples:
        + Distribute a 24 hour 100 mm rainstorm in a 12 hour gaussian pulse
        -> storm = RainStorm('norm')
        -> storm = storm.compute(timestep=0.5, duration=12, rainfall=100)
        -> storm.pr.plot()

        + Create a 24 hour storm following the SCS type I hyetograph with 
        + pulses every 10 minutes and a total precipitation of 75 mm.
        + Then compute infiltration using SCS method and a basin CN of 75
        -> storm = RainStorm('SCS_I24')
        -> storm = storm.compute(timestep=10/60, duration=24, rainfall=75)
        -> storm = storm.infiltrate(method='SCS', cn=75)
        -> storm.pr.plot()
        -> storm.losses.plot()

        + Create a narrow and wide gaussian pulse of 100 mm in 12 hours
        -> narrow = RainStorm('norm', loc=0.5, scale=0.05)
        -> wide   = RainStorm('norm', loc=0.5, scale=0.15)
        -> narrow = storm.compute(timestep=0.5, duration=12, rainfall=100)
        -> wide   = storm.compute(timestep=0.5, duration=12, rainfall=100)
    """

    def synth_rain(self,
                   loc: float, scale: float, flip: bool = False, n: int = 1000,
                   **kwargs: Any) -> pd.Series:
        """
        Synthetic hyetograph generator function. If the storm type given
        in the class constructor is part of any of scipy distributions 
        the synthetic hyetograph will be built with the given loc, scale
        and scipy default parameters. 

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
        kind = self.kind
        scipy_distrs = [d for d in dir(st)
                        if isinstance(getattr(st, d), st.rv_continuous)]
        if kind in scipy_distrs:
            distr = eval(f'st.{kind}')
            shyeto = distr.pdf(time_dimless, loc=loc, scale=scale,
                               **kwargs)
            shyeto = shyeto/np.sum(shyeto)
            if flip:
                shyeto = pd.Series(shyeto[::-1], index=time_dimless)
            else:
                shyeto = pd.Series(shyeto, index=time_dimless)
        else:
            shyeto = SHYETO_DATA[kind]
        return shyeto

    def __init__(self, kind: str = 'norm', loc: float = 0.5, scale: float = 0.1,
                 **kwargs: Any) -> None:
        """
        Synthetic RainStorm builder

        Args:
            kind (str): Type of synthetic hyetograph to use. It can be of two
                types:
                    1) Any of scipy distributiosn (give parameters in **kwargs)
                    2) Any of
                        "SCS_X" with X = I24,IA24,II6,II12,II24,II48,III24
                        "GX_Benitez1985" with X = 1,2,3
                        "GX_Espildora1979" with X = 1,2,3
                        "GXpY_Varas1985" with X = 1,2,3,4 and Y=10,25,50,75,90
                Defaults to 'norm'.
            loc (float): Number between 0 - 1 to specify location parameter
                for statistic-like rainfall distribution. Defaults to 0.5.
            scale (float): Number between 0 -1 to specify scale parameter
                for statistic-like rainfall distribution. Defaults to 0.1.
            **kwargs are given to scipy.rv_continuous.pdf

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
        self.losses = None
        self.pr_dimless = self.synth_rain(loc=loc, scale=scale, **kwargs)

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

    def compute(self, timestep: Union[int, float], duration: Union[int, float],
                rainfall: ArrayLike, n: int = 1,
                interp_kwargs: dict = {'method': 'linear'}
                ) -> Type['RainStorm']:
        """
        Trigger computation of design storm for a given timestep, storm 
        duration, and total precipitation.

        Args:
            timestep (float): Storm timestep or resolution in hours
            duration (float): Storm duration in hours
            rainfall (array_like or float): Total precipitation in mm. 
            n (int, optional): If n=1 the storm time length will be equal to 
                the user storm duration. If n>1 it will fill the time index with
                n zeros. Defaults to 1.
            interp_kwargs (dict): extra arguments for the interpolation function

        Returns:
            Updated class
        """
        self.timestep = timestep
        self.duration = duration
        self.rainfall = rainfall

        xr_rainfall = obj_to_xarray(rainfall).squeeze()
        dims = {dim: xr_rainfall[dim].shape[0] for dim in xr_rainfall.dims}
        time1 = np.arange(0, duration, timestep)
        time2 = np.arange(0, duration+timestep, timestep)
        time3 = np.arange(0, duration+n*timestep, timestep)

        # Build dimensionless storm (accumulates 1 mm)
        shyeto = obj_to_xarray(self.pr_dimless.cumsum(), dims=('time'),
                               coords={'time': self.pr_dimless.index})
        shyeto = shyeto.interp(coords={'time': np.linspace(0, 1, len(time1))},
                               **interp_kwargs)
        shyeto.coords['time'] = time1

        # Build real storm for the given rainfall
        storm = shyeto.expand_dims(dim=dims)*xr_rainfall
        storm = storm.reindex({'time': time2}).shift({'time': 1})
        storm = storm.diff('time').transpose(*(['time']+list(dims.keys())))
        storm = storm.where(storm >= 0).fillna(0)
        storm = storm.reindex({'time': time3}).fillna(0)
        storm.name = 'pr'

        self.pr = storm
        return self

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
        self.infiltration = method
        storm = self.pr
        if method == 'SCS':
            cn = kwargs['cn']
            kwargs = kwargs.copy()
            kwargs.pop('cn', None)

            storm_cum = storm.cumsum('time')
            time = np.arange(0, self.duration+self.timestep, self.timestep)
            losses = xr.apply_ufunc(SCS_Abstractions, storm_cum, cn,
                                    kwargs=kwargs,
                                    input_core_dims=[['time'], []],
                                    output_core_dims=[['time']],
                                    vectorize=True)
            losses = losses.reindex({'time': time})
            losses = losses.transpose(*storm.dims).diff('time')
            losses = losses.where(losses >= 0).fillna(0)

            pr_eff = self.pr-losses
            pr_eff = pr_eff.where(pr_eff >= 0).fillna(0)

            self.losses = losses
            self.pr_eff = pr_eff
        else:
            raise ValueError(f'{method} unknown infiltration method.')
        return self
