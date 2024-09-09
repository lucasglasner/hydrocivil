'''
 # @ Author: Lucas Glasner (lgvivanco96@gmail.com)
 # @ Create Time: 1969-12-31 21:00:00
 # @ Modified by: Lucas Glasner,
 # @ Modified time: 2024-05-06 16:40:13
 # @ Description:
 # @ Dependencies:
 '''

import pandas as pd
import numpy as np
# ------------------------ SCS curve number equations ------------------------ #


def cn_correction(cn_II, amc):
    """
    This function changes the curve number value according to antecedent
    moisture conditions (amc)...

    Reference:
        Ven Te Chow (1988), Applied Hydrology. MCGrow-Hill
        Soil Conservation Service, Urban hydrology for small watersheds,
        tech. re/. No. 55, U. S. Dept. of Agriculture, Washington, D.E:., 1975.

    Args:
        cn_II (float): curve number under normal condition
        amc (str): Antecedent moisture condition.
            Options: 'dry', 'wet' or 'normal'

    Raises:
        RuntimeError: If amc is different than 'dry', 'wet' or 'normal'

    Returns:
        cn_I or cn_III (float): _description_
    """
    if (amc == 'dry') or (amc == 'I'):
        cn_I = 4.2*cn_II/(10-0.058*cn_II)
        return cn_I
    elif (amc == 'normal') or (amc == 'II'):
        return cn_II
    elif (amc == 'wet') or (amc == 'III'):
        cn_III = 23*cn_II/(10+0.13*cn_II)
        return cn_III
    else:
        text = f'amc="{amc}"'
        text = text+' Unkown antecedent moisture condition.'
        raise RuntimeError(text)


def SCS_MaximumRetention(cn, cfactor=25.4):
    """
    Simple function for the SCS maximum potential retention of the soil

    Args:
        cn (float): Curve number (dimensionless)
        cfactor (float): Unit conversion factor.
            cfactor = 1 --> inches
            cfactor = 25.4 --> milimeters

    Returns:
        (float): maximum soil retention in mm by default
    """
    S = 1000/cn - 10
    return cfactor*S


def SCS_EffectiveRainfall(pr, cn, r=0.2, amc='II', weights=None, **kwargs):
    """
    SCS formula for overall effective precipitation/runoff.

    Args:
        pr (array_like): Precipitation in mm
        cn (array_like or float): Curve Number
        r (float, optional): Fraction of the maximum potential retention
            Defaults to 0.2.
        amc (str): Antecedent moisture condition. Defaults to 'II'. 
            Options: 'dry' or 'I',
                     'normal' or 'II'
                     'wet' or 'III'
        weights (array_like). If curve number is an array of values this
            attribute expects an array of the same size with weights for
            the precipitation computation. Defaults to None.
        **kwargs are passed to SCS_MaximumRetention function

    Returns:
        (array_like): Effective precipitation (Precipitation - Infiltration)
    """
    cn = cn_correction(cn, amc)
    if not np.isscalar(cn):
        if type(weights) != type(None):
            pr_eff = [SCS_EffectiveRainfall(pr, cn_i, r, weights=None)*w
                      for cn_i, w in zip(cn, weights)]
            if np.isscalar(pr):
                pr_eff = np.sum(pr_eff)
            else:
                pr_eff = pd.concat(pr_eff, keys=cn).unstack(1).sum(axis=0)
        else:
            raise ValueError(
                'If cn is iterable must give weights to each land cover class!')
    else:
        S = SCS_MaximumRetention(cn, **kwargs)
        Ia = r*S
        pr_eff = (pr-Ia)**2/(pr-Ia+S)
        if np.isscalar(pr):
            if pr <= Ia:
                pr_eff = 0
        else:
            pr_eff[pr <= Ia] = 0
    return pr_eff


def SCS_Abstraction(pr, cn, r=0.2, **kwargs):
    """
    SCS formula for overall water losses due to infiltration/abstraction

    Args:
        pr (array_like): Precipitation in mm 
        cn (float): Curve Number
        r (float, optional): Fraction of the maximum potential retention
            Defaults to 0.2.
        **kwargs are passed to SCS_MaximumRetention function


    Returns:
        (array_like): Infiltration
    """
    pr_eff = SCS_EffectiveRainfall(pr, cn, r, **kwargs)
    inf = pr-pr_eff
    return inf
