'''
 # @ Author: Lucas Glasner (lgvivanco96@gmail.com)
 # @ Create Time: 1969-12-31 21:00:00
 # @ Modified by: Lucas Glasner,
 # @ Modified time: 2024-05-06 16:24:28
 # @ Description:
 # @ Dependencies:
 '''

import os
import numpy as np
import pandas as pd
import xarray as xr
import warnings

from src.infiltration import SCS_Losses
from scipy.interpolate import interp1d
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


class DesignStorm(object):
    def __init__(self, kind):
        """
        Storm design class

        Args:
            kind (str): Type of synthetic hyetograph to use:
                Options:
                    'UNIFORM'
                    'G1_Espildora1979'
                    'G2_Espildora1979'
                    'G3_Espildora1979'
                    'G1_Benitez1985'
                    'G2_Benitez1985'
                    'G3_Benitez1985'
                    'G1_Varas1985'
                    'G2_Varas1985'
                    'G3_Varas1985'
                    'G4_Varas1985'
        """
        shyeto_path = os.path.join('data', 'synthethic_storms.csv')
        shyeto = pd.read_csv(shyeto_path, index_col=0,
                             usecols=['time', kind])[kind]
        self.SynthHyetograph = shyeto
        self.kind = kind

        self.timestep = None
        self.duration = None
        self.rainfall = None
        self.ref_duration = None
        self.Hyetograph = None
        self.Effective_Hyetograph = None
        self.Losses = None

    def __repr__(self) -> str:
        """
        What to show when invoking a DesignStorm object
        Returns:
            str: Some metadata
        """
        text = f'Storm type: {self.kind}\n'
        if type(self.Hyetograph) != type(None):
            text = text+f'Total rainfall:\n{self.Hyetograph.sum(axis=0)}\n'
        if type(self.Losses) != type(None):
            text = text+f'Total losses:\n{self.Losses.sum(axis=0)}\n'
        return text

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
            losses = SCS_Losses(storm_cum, **kwargs)
            self.Losses = losses.diff().fillna(0)
            self.Effective_Hyetograph = self.Hyetograph-self.Losses
        else:
            raise ValueError(f'{method} unknown infiltration method.')
        return self

    def compute(self, timestep, duration, rainfall, ref_duration=24, **kwargs):
        """
        Trigger computation of design storm for a given timestep, storm 
        duration, and precipitation for a reference storm (pr and ref_duration)

        Args:
            timestep (float): Storm timestep or resolution in hours
            duration (float): Total storm duration in hours
            rainfall (1D array_like or float): Total precipitation in mm
            ref_duration (float): Duration of the given precipitation.
                Defaults to 24h.
            **kwargs are given to the interpolation function

        Returns:
            Updated class
        """
        self.timestep = timestep
        self.duration = duration
        self.rainfall = rainfall
        self.ref_duration = ref_duration
        shyeto = self.SynthHyetograph
        func = interp1d(shyeto.index*duration, shyeto.values,
                        fill_value='extrapolate', **kwargs)
        time = np.arange(0, duration+timestep, timestep)
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
        return self

    # def plot(self, Losses_kwargs={}, **kwargs):
    #     """
    #     Plot a simple time vs rain graph

    #     Raises:
    #         RuntimeError: If a Hyetograph isnt already computed
    #     """
    #     if type(self.Hyetograph) != type(None):
    #         axes = self.Hyetograph.plot(**kwargs)
    #         axes = axes.axes
    #     else:
    #         raise RuntimeError('Compute a Hyetograph before plotting!')
    #     if type(self.Losses) != type(None):
    #         self.Losses.plot(ax=axes, color='tab:red', legend=False,
    #                          **Losses_kwargs)
