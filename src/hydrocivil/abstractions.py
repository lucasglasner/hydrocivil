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


def SCS_EquivalentCurveNumber(pr, pr_eff, r=0.2, cfactor=25.4):
    """
    Given a rainfall ammount and the related effective precipitation (surface
    runoff) observed, this function computes the equivalent curve number of the
    soil. 

    Args:
        pr (1D array_like or float): Precipitation in mm
        pr_eff (1D array_like or float): Effective precipitation in mm
        r (float, optional): Fraction of the maximum potential retention
            used on the initial abstraction calculation. Defaults to 0.2.
        cfactor (float): Unit conversion factor.
            cfactor = 1 --> inches
            cfactor = 25.4 --> milimeters

    Reference:
        Stowhas, Ludwig (2003). Uso del método de la curva número en cuencas
        heterogéneas. XVI Congreso de la Sociedad Chilena de Ingeniería
        Hidráulica. Pontificia Universidad Católica de Chile. Santiago, Chile.

    Returns:
        (1D array_like or float): Equivalent curve number
    """
    a = r**2
    b = -2*r*pr-pr_eff*(1-r)
    c = pr**2-pr_eff*pr
    S_eq = (-b-(b**2-4*a*c)**0.5)/2/a
    CN_eq = 1000*cfactor/(S_eq+10*cfactor)
    return CN_eq


def SCS_EffectiveRainfall(pr, cn, r=0.2, weights=None, **kwargs):
    """
    SCS formula for overall effective precipitation/runoff. Function
    adapted to work for scalar inputs or array_like inputs. If you give
    weights the program will try to compute the curvenumber-weighted 
    runoff. If not the program will compute runoff with the user cn and pr. 
    If pr and cn are arrays they must have the same size for index-wise
    computations.  

    Args:
        pr (1D array_like or float): Precipitation in mm
        cn (1D array_like or float): Curve Number
        r (float, optional): Fraction of the maximum potential retention
            used on the initial abstraction calculation. Defaults to 0.2.
        weights (array_like or None). If curve number is an array of values this
            attribute expects an array of the same size with areal weights for
            the precipitation computation. Defaults to None.
        **kwargs are passed to SCS_MaximumRetention function

    Returns:
        (array_like): Effective precipitation (Precipitation - Infiltration)
    """
    if np.isscalar(pr):
        return SCS_EffectiveRainfall(np.array([pr]), cn, r, weights, **kwargs)

    if isinstance(pr, pd.Series) or isinstance(pr, pd.DataFrame):
        pr = pr.values

    if np.isscalar(cn):
        cn = np.full(pr.shape, cn)
        return SCS_EffectiveRainfall(pr, cn, r, weights, **kwargs)

    if (type(weights) != type(None)):
        weights = np.stack([weights]*len(pr))
        S = SCS_MaximumRetention(cn, **kwargs)
        S = np.stack([S]*len(pr)).T
        Ia = r*S

        pr = np.stack([pr]*len(cn))
        pr_eff = (pr-Ia)**2/(pr-Ia+S)
        # pr_eff = np.where(pr<=Ia, 0, pr_eff)
        pr_eff[pr <= Ia] = 0
        pr_eff = np.stack(weights*pr_eff.T).sum(axis=-1)
    elif (cn.shape == pr.shape):
        S = SCS_MaximumRetention(cn, **kwargs)
        Ia = r*S
        pr_eff = (pr-Ia)**2/(pr-Ia+S)
        # pr_eff = np.where(pr<=Ia, 0, pr_eff)
        pr_eff[pr <= Ia] = 0
    else:
        text = 'pr and cn must have the same size for index-wise computation.'
        text = text+' If not you must give weights so the program can compute '
        text = text+'the curvenumber-weighted average.'
        raise RuntimeError(text)
    return pr_eff


def SCS_Abstractions(pr, cn, r=0.2, weights=None, **kwargs):
    """
    SCS formula for overall water losses due to infiltration/abstraction.
    Losses are computed simply as total precipitation - total runoff. 

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
    pr_eff = SCS_EffectiveRainfall(pr, cn, r=r, weights=weights, **kwargs)
    Losses = pr-pr_eff
    return Losses
