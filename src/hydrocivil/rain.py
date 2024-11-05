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

from .abstractions import SCS_Abstractions
from .global_vars import SHYETO_DATA

from scipy.interpolate import interp1d
import scipy.stats as st

# ----------------------- duration coefficient routines ---------------------- #


def grunsky_coef(storm_duration, ref_duration=24):
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


def bell_coef(storm_duration, ref_duration=24):
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


def duration_coef(storm_duration,
                  ref_duration=24,
                  bell_threshold=1,
                  duration_threshold=10/60):
    """
    This function is a merge of Grunsky and Bell Formulations
    of the Duration Coefficient. The idea is to use Bell's
    Formula only when the input duration is less than the "bell_threshold"
    parameter. In addition, when the duration is less than the
    "duration_threshold" the duration is set to the "duration_threshold".

    Args:
        storm_duration (array_like): Storm duration in hours
        bell_threshold (float, optional): Duration threshold for changing
            between Grunsky and Bell formulas. Defaults to 1 (hour).
        duration_threshold (float, optional): Minimum storm duration.
            Defaults to 10 minutes (1/6 hours).

    Returns:
        coef (array_like): Duration coefficients (dimensionless)
    """
    t = storm_duration
    if not np.isscalar(t):
        coefs = np.empty(len(t))
        for i in range(len(coefs)):
            if t[i] < duration_threshold:
                coefs[i] = duration_coef(duration_threshold, ref_duration)
                text = f'Storm duration is less than {duration_threshold}'
                text = text+f' threshold, changing to {duration_threshold}.'
                warnings.warn(text)
            elif (t[i] >= duration_threshold) and (t[i] < bell_threshold):
                coefs[i] = bell_coef(t[i], ref_duration)
            else:
                coefs[i] = grunsky_coef(t[i], ref_duration)
    else:
        coefs = duration_coef([t], ref_duration, bell_threshold)
        coefs = coefs.item()
    return coefs


# ------------------------------- Design Storms ------------------------------ #


class RainStorm(object):
    """
    RainStorm class used to building temporal rainfall distributions. 
    The class can be used to build rainstorms that follow any of scipy
    theoretical distributions (e.g 'norm', 'skewnorm', 'gamma', etc) or 
    the empirical rain distributions of the SCS type I, IA, II, III and the 
    Chilean synthetic hyetographs of (Espildora et al 1979),
    (Benitez et al 1985) and (Varas et al 1985). 

    Examples:
        #### Distribute a 24 hour 100 mm rainstorm in a 12 hour gaussian pulse
        -> storm = RainStorm('norm')
        -> storm = storm.compute(timestep=0.5, duration=12, rainfall=100)
        -> storm.Hyetograph.plot()

        #### Create a 24 hour storm following the SCS type I hyetograph with 
        #### pulses every 10 minutes and a total precipitation of 75 mm.
        #### Then compute infiltration using SCS method and a basin CN of 75
        -> storm = RainStorm('SCS_I24')
        -> storm = storm.compute(timestep=10/60, duration=24, rainfall=75)
        -> storm = storm.infiltrate(method='SCS', cn=75)
        -> storm.Hyetograph.plot()
        -> storm.Losses.plot()

        #### Create a narrow and wide gaussian pulse of 100 mm in 12 hours
        -> narrow = RainStorm('norm', loc=0.5, scale=0.05)
        -> wide   = RainStorm('norm', loc=0.5, scale=0.15)
        -> narrow = storm.compute(timestep=0.5, duration=12, rainfall=100,
                                  ref_duration=12)
        -> wide   = storm.compute(timestep=0.5, duration=12, rainfall=100,
                                  ref_duration=12)
    """

    def synthetic_hyetograph(self, loc, scale, flip=False, **kwargs):
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
            **kwargs are given to scipy.rv_continuous.pdf

        Returns:
            (pandas.Series): Synthetic Hyetograph 1D Table
        """
        kind = self.kind
        scipy_distrs = [d for d in dir(st)
                        if isinstance(getattr(st, d), st.rv_continuous)]
        if kind in scipy_distrs:
            distr = eval(f'st.{kind}')
            shyeto = distr.pdf(np.linspace(0, 1), loc=loc, scale=scale,
                               **kwargs)
            shyeto = shyeto/np.sum(shyeto)
            if flip:
                shyeto = pd.Series(shyeto[::-1], index=np.linspace(0, 1))
            else:
                shyeto = pd.Series(shyeto, index=np.linspace(0, 1))
        else:
            shyeto = SHYETO_DATA[kind]
        return shyeto

    def __init__(self, kind, loc=0.5, scale=0.1, **kwargs):
        """
        Synthetic RainStorm builder

        Args:
            kind (str): Type of synthetic hyetograph to use
            loc (float): Number between 0 - 1 to specify location parameter
                for statistic-like rainfall distribution. Defaults to 0.5.
            scale (float): Number between 0 -1 to specify scale parameter
                for statistic-like rainfall distribution. Defaults to 0.1.
            **kwargs are given to scipy.rv_continuous.pdf
        """

        self.kind = kind
        self.timestep = None
        self.duration = None
        self.rainfall = None
        self.ref_duration = None
        self.Hyetograph = None
        self.Effective_Hyetograph = None
        self.Losses = None

        self.SynthHyeto = self.synthetic_hyetograph(loc=loc, scale=scale,
                                                    **kwargs)

    def __repr__(self) -> str:
        """
        What to show when invoking a RainStorm object
        Returns:
            str: Some metadata
        """
        text = f'Storm type: {self.kind}\n'
        if type(self.Hyetograph) != type(None):
            text = text+f'Total rainfall:\n{self.Hyetograph.sum(axis=0)}\n'
        if type(self.Losses) != type(None):
            text = text+f'Total losses:\n{self.Losses.sum(axis=0)}\n'
        return text

    def copy(self):
        """
        Create a deep copy of the class itself
        """
        return pycopy.deepcopy(self)

    def infiltrate(self, method='SCS', **kwargs):
        """
        Compute losses due to infiltration with different methods for the
        stored storm Hyetograph
        Args:
            method (str, optional): Infiltration routine. Defaults to 'SCS'.

        Returns:
            Updated class
        """
        storm = self.Hyetograph
        if method == 'SCS':
            storm_cum = storm.cumsum()
            losses = SCS_Abstractions(storm_cum, **kwargs)
            self.Losses = losses.diff().fillna(0)
            self.Effective_Hyetograph = self.Hyetograph-self.Losses
        else:
            raise ValueError(f'{method} unknown infiltration method.')
        return self

    def compute(self, timestep, duration, rainfall, ref_duration=24,
                interp_kwargs={}, **kwargs):
        """
        Trigger computation of design storm for a given timestep, storm 
        duration, and precipitation for a reference storm (pr and ref_duration)

        Args:
            timestep (float): Storm timestep or resolution in hours
            duration (float): Total storm duration in hours
            rainfall (1D array_like or float): Total precipitation in mm. 
                Usually a function of the return period. 
            ref_duration (float): Duration of the given reference precipitation.
                Defaults to 24h.
            interp_kwargs (dict): extra arguments for the interpolation function
            **kwargs are given to the synthetic hyetograph generator

        Returns:
            Updated class
        """
        self.timestep = timestep
        self.duration = duration
        self.rainfall = rainfall
        self.ref_duration = ref_duration
        time = np.arange(0, duration+timestep, timestep)

        func = interp1d(self.SynthHyeto.index*duration, self.SynthHyeto.values,
                        fill_value='extrapolate', **interp_kwargs)
        storm = pd.Series(func(time), index=time).cumsum()

        if np.isscalar(rainfall):
            pr_fix = rainfall*duration_coef(duration, ref_duration)
            storm = (storm*pr_fix/storm.max()).diff().fillna(0)
        else:
            if not isinstance(rainfall, pd.Series):
                rainfall = pd.Series(rainfall)
            storm = [(storm*p*duration_coef(duration, ref_duration) /
                      storm.max()).diff().fillna(0) for p in rainfall]
            storm = pd.concat(storm, axis=1)
            storm.columns = rainfall.index
        self.Hyetograph = storm
        return self.copy()

    def plot(self, legend=True, **kwargs):
        """
        Plot a simple time vs rain graph

        Raises:
            RuntimeError: If a Hyetograph isnt already computed
        """
        if type(self.Hyetograph) != type(None):
            axes = self.Hyetograph.plot(kind='bar', width=1, color='tab:blue',
                                        label='Rainfall',
                                        **kwargs)
            axes = axes.axes
        else:
            raise RuntimeError('Compute a Hyetograph before plotting!')
        if type(self.Losses) != type(None):
            self.Losses.plot(ax=axes, kind='bar', width=1, color='tab:purple',
                             label='Abstractions',
                             legend=False, **kwargs)
        xticks = np.arange(0, self.duration/self.timestep+1, 1)
        n = int(len(xticks)/self.duration)
        if n != 0:
            axes.set_xticks(xticks[::n])
        else:
            axes.set_xticks(xticks)
        return axes
