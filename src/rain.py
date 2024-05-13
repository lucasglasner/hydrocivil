'''
 # @ Author: Lucas Glasner (lgvivanco96@gmail.com)
 # @ Create Time: 1969-12-31 21:00:00
 # @ Modified by: Lucas Glasner, 
 # @ Modified time: 2024-05-06 16:24:28
 # @ Description:
 # @ Dependencies:
 '''

import numpy as np
import pandas as pd
import xarray as xr

from src.misc import is_iterable
from src.infiltration import cum_abstraction
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
        Bell, F.C. (1969) Generalized Rainfall-Duration-Frequency
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
    if is_iterable(t):
        coefs = np.empty(len(t))
        for i in range(len(coefs)):
            if t[i] < duration_threshold:
                coefs[i] = duration_coef(duration_threshold, ref_duration)
            elif (t[i] >= duration_threshold) and (t[i] < bell_threshold):
                coefs[i] = bell_coef(t[i], ref_duration)
            else:
                coefs[i] = grunsky_coef(t[i], ref_duration)
    else:
        coefs = duration_coef([t], ref_duration, bell_threshold)
        coefs = coefs.item()
    return coefs


# ------------------------------- Design Storms ------------------------------ #
def effective_rainfall(pr, duration, CN):
    """
    SCS formula for overall effective precipitation. 

    Args:
        pr (array_like): Precipitations. 
        duration (float): Design storm duration
        CN (float): Curve Number

    Returns:
        (array_like): Effective precipitation (Precipitation - Infiltration)
    """
    # S = maximum infiltration given soil type and cover (CN)
    # I0 = 0.2 * S infiltration until pond formation
    S = (25400-254*CN)/CN  # SCS maximum infiltration capacity
    pr_t = pr*duration_coef(duration)
    pr_eff = (pr_t-0.2*S)*(pr_t-0.2*S)/(pr_t+0.8*S)
    return pr_eff


def design_storms(storm_duration, synth_hyeto, tstep, precips,
                  interp_kwargs={'kind': 'quadratic'}):
    """
    Function to generate design storms for a given storm duration, 
    synthetic hyetograph (hyetograph shape), time step and total accumulated
    precipitation in 24 hrs. 

    Args:
        storm_duration (float): Storm duration in hours. 
        synth_hyeto (pandas.DataFrame): Storm shape.
        tstep (float): Model timestep in hours
        precips (pandas.Series): precipitation data in mm / day. 
        interp_kwargs (dict): args to scipy.interpolation.interp1d function.

    Returns:
        (xr.DataArray): xarray with hyetographs
    """
    if is_iterable(storm_duration):
        storms = [design_storms(dt, synth_hyeto, tstep, precips, interp_kwargs)
                  for dt in storm_duration]
        storms = xr.concat(storms, 'storm_duration')
        storms.coords['storm_duration'] = (
            'storm_duration', storm_duration, {'units': 'hours'})
    else:
        # Definitions
        new_time = np.arange(0, storm_duration+tstep, tstep)
        returnperiods = precips.index
        storms_names = synth_hyeto.columns

        # Compute accumulated precipitation for the overall event
        # Set the new time index based on the storm duration
        acc_storms = synth_hyeto.cumsum().copy()
        acc_storms.index = acc_storms.index*storm_duration

        # Interpolate to the new time step
        interpfuncs = [interp1d(acc_storms.index, acc_storms[c].values,
                                fill_value='extrapolate', **interp_kwargs)
                       for c in storms_names]
        acc_storms = [pd.Series(f(new_time), index=new_time, name=name)
                      for f, name in zip(interpfuncs, storms_names)]
        acc_storms = [storm/storm.sum()
                      for storm in acc_storms]  # Force accum(Pr) = 1
        acc_storms = pd.concat(acc_storms, axis=1)
        acc_storms.index = np.round(acc_storms.index, 4)

        # Asign precipitations to the synthetic storms
        rainfall = pd.concat([precips]*len(acc_storms.index),
                             keys=acc_storms.index).swaplevel()
        rainfall = pd.concat([rainfall]*len(acc_storms.columns),
                             keys=acc_storms.columns).unstack(level=0)
        acc_storms = pd.concat([rainfall.loc[T]*acc_storms
                                for T in returnperiods],
                               keys=returnperiods)

        # Return as an n-dimensional array of non-accumulated rain
        storms = [acc_storms.loc[T].diff(axis=0).bfill()
                  for T in returnperiods]
        storms = [s/s.sum()*rainfall.loc[T]
                  for s, T in zip(storms, returnperiods)]
        storms = xr.DataArray(storms, coords={'return_period': returnperiods,
                                              'time': new_time,
                                              'shyeto': storms_names},
                              attrs={'standard_name': 'precipitation rate',
                                     'units': 'mm*day-1'})
        storms = storms.expand_dims(
            dim={'storm_duration': [storm_duration]}, axis=0)
        storms.coords['return_period'].attrs = {'standard_name': 'return_period',
                                                'units': 'years'}
        storms.coords['time'].attrs = {'standard_name': 'simulation time',
                                       'units': 'hours'}
        storms.coords['shyeto'].attrs = {
            'standard_name': 'synthetic hyetograph'}
        storms = storms.to_dataset(name='pr')
    return storms


def effective_storms(storms, method='SCS', **kwargs):
    """
    This function compute the effective precipitation for a given storm.
    The basic idea is to compute the net infiltration along time and
    substract it from the total precipitation. 

    Args:
        storms (xr.DataArray): xarray with hyetographs
        method (str, optional): Method to compute infiltration.
            Defaults to 'SCS'.
        **kwargs are given to the infiltration routine

    Returns:
        (xr.DataArray): xarray with effective hyetographs
    """
    if method == 'SCS':
        attrs = storms.attrs
        dcoef = [duration_coef(dt) for dt in storms.storm_duration]
        dcoef = xr.DataArray(
            dcoef, coords={'storm_duration': storms.storm_duration})
        pr_eff = storms.cumsum('time')*dcoef
        pr_eff.name = 'pr_eff'
        infilt = pr_eff.to_dataframe().map(cum_abstraction, **kwargs)
        pr_eff = pr_eff-infilt.to_xarray()
        pr_eff = pr_eff.diff('time').reindex(
            {'time': storms.time}).bfill('time')
        pr_eff = pr_eff.where(storms >= 0)
        storms = xr.merge([storms, pr_eff])
        storms.pr_eff.attrs = attrs
        storms.pr_eff.attrs['standard_name'] = 'effective precipitation rate'
    return storms
