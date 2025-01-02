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
from numpy.typing import ArrayLike
from typing import Union
# ------------------------ SCS curve number equations ------------------------ #


def cn_correction(cn_II: Union[int, float, ArrayLike],
                  amc: str) -> Union[float, ArrayLike]:
    """
    This function changes the curve number value according to antecedent
    moisture conditions (amc)...

    Args:
        cn_II (int|float|ArrayLike): curve number under normal condition
        amc (str): Antecedent moisture condition.
            Options: 'dry'|'I', 'wet'|'III' or 'normal'|'II'

    Raises:
        RuntimeError: If amc is different than 'dry', 'I', 'wet', 'III' or
            'normal', 'II'. 

    Returns:
        (float): adjusted curve number for given AMC

    Reference:
        Ven Te Chow (1988), Applied Hydrology. MCGrow-Hill
        Soil Conservation Service, Urban hydrology for small watersheds,
        tech. re/. No. 55, U. S. Dept. of Agriculture, Washington, D.E:., 1975.
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


def SCS_MaximumRetention(cn: Union[int, float, ArrayLike],
                         cfactor: float = 25.4) -> Union[float, ArrayLike]:
    """
    Calculate SCS soil maximum potential retention.

    Args:
        cn (int|float|ArrayLike): Curve number (dimensionless)
        cfactor (float): Unit conversion factor.
            cfactor = 1 --> inches
            cfactor = 25.4 --> milimeters

    Returns:
        (float): maximum soil retention in mm by default
    """
    S = 1000/cn - 10
    return cfactor*S


def SCS_EquivalentCurveNumber(pr: Union[int, float, ArrayLike],
                              pr_eff: Union[int, float, ArrayLike],
                              r: float = 0.2,
                              cfactor: float = 25.4
                              ) -> Union[float, ArrayLike]:
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


def SCS_EffectiveRainfall(pr: Union[int, float, ArrayLike],
                          cn: Union[int, float, ArrayLike],
                          r: float = 0.2,
                          **kwargs: float) -> Union[float, ArrayLike]:
    """
    SCS formula for effective precipitation/runoff.

    Args:
        pr (float|array): Precipitation depth [mm]
        cn (float|array): Curve Number [-]
        r (float): Initial abstraction ratio, default 0.2

    Returns:
        float|array: Effective rainfall depth [mm]

    Examples:
        >>> SCS_EffectiveRainfall(50, 80)
        22.89
        >>> SCS_EffectiveRainfall([10,20,30], 75)
        array([0., 2.45, 8.67])
    """
    def _scalar(pr: Union[int, float],
                cn: Union[int, float],
                r: float = 0.2,
                **kwargs: float) -> Union[float]:
        """
        Core SCS calculation for scalar inputs

        Raises:
            ValueError: If pr is not positive
            ValueError: If CN is not between 0 and 100
            ValueError: If r is not positive

        Returns:
            float: Effective rainfall depth [mm]
        """
        if np.isnan(pr) or np.isnan(cn):
            return np.nan
        if pr < 0:
            raise ValueError("Precipitation must be positive")
        if not 0 <= cn <= 100:
            raise ValueError("CN must be between 0 and 100")
        if r <= 0:
            raise ValueError("Initial abstraction ratio must be positive")
        S = SCS_MaximumRetention(cn, **kwargs)
        Ia = r*S
        if pr <= Ia:
            pr_eff = 0
        else:
            pr_eff = (pr-Ia)**2/(pr-Ia+S)
        return pr_eff

    if np.isscalar(pr) and np.isscalar(cn):
        return _scalar(pr, cn, r, **kwargs)
    else:
        return np.vectorize(_scalar, otypes=[float])(pr, cn, r, **kwargs)


def SCS_Abstractions(pr: Union[int, float, ArrayLike],
                     cn: Union[int, float, ArrayLike],
                     r: float = 0.2,
                     **kwargs: float) -> Union[float, ArrayLike]:
    """
    SCS formula for overall water losses due to infiltration/abstraction.
    Losses are computed simply as total precipitation - total runoff. 

    Args:
        pr (array_like or float): Precipitation in mm 
        cn (array_like or float): Curve Number
        r (float, optional): Fraction of the maximum potential retention
            Defaults to 0.2.
        **kwargs are passed to SCS_MaximumRetention function


    Returns:
        (array_like): Losses/Abstraction/Infiltration
    """
    pr_eff = SCS_EffectiveRainfall(pr, cn, r=r, **kwargs)
    Losses = pr-pr_eff
    return Losses
