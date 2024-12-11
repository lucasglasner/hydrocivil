'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 1969-12-31 21:00:00
 Modified by: Lucas Glasner,
 Modified time: 2024-05-06 16:40:13
 Description:
 Dependencies:
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


def SCS_EffectiveRainfall(pr, cn, r=0.2, weights=None, **kwargs):
    """
    SCS formula for overall effective precipitation/runoff. Function
    adapted to work for scalar inputs or array_like inputs.

    Args:
        pr (array_like or float): Precipitation in mm
        cn (array_like or float): Curve Number
        r (float, optional): Fraction of the maximum potential retention
            used on the initial abstraction calculation. Defaults to 0.2.
        weights (array_like or None). If curve number is an array of values this
            attribute expects an array of the same size with weights for
            the precipitation computation. Defaults to None.
        **kwargs are passed to SCS_MaximumRetention function

    Returns:
        (array_like): Effective precipitation (Precipitation - Infiltration)
    """
    if np.isscalar(pr):
        return SCS_EffectiveRainfall([pr], cn, r, weights, **kwargs)
    else:
        if np.isscalar(cn):
            S = SCS_MaximumRetention(cn, **kwargs)
            Ia = r*S
            pr_eff = (pr-Ia)**2/(pr-Ia+S)
            pr_eff[pr <= Ia] = 0
        else:
            if type(weights) != type(None):
                pr_eff = [w*SCS_EffectiveRainfall(pr, cn_i, r, None, **kwargs)
                          for cn_i, w in zip(cn, weights)]
                pr_eff = sum(pr_eff)
            else:
                text = 'Weights must be added for each land class!'
                raise ValueError(text)
        return pr_eff


def SCS_Abstractions(pr, cn, r=0.2, weights=None, **kwargs):
    """
    SCS formula for overall water losses due to infiltration/abstraction

    Args:
        pr (array_like or float): Precipitation in mm 
        cn (array_like or float): Curve Number
        r (float, optional): Fraction of the maximum potential retention
            Defaults to 0.2.
        weights (array_like or None). If curve number is an array of values this
            attribute expects an array of the same size with weights for
            the precipitation computation. Defaults to None.
        **kwargs are passed to SCS_MaximumRetention function


    Returns:
        (array_like): Losses/Abstraction/Infiltration
    """
    pr_eff = SCS_EffectiveRainfall(pr, cn, r, weights, **kwargs)
    Losses = pr-pr_eff
    return Losses
