'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 1969-12-31 21:00:00
 Modified by: Lucas Glasner,
 Modified time: 2024-05-06 16:40:13
 Description:
 Dependencies:
'''

from scipy.optimize import root_scalar
import numpy as np
from numpy.typing import ArrayLike


# --------------------------- Vadose zone equations -------------------------- #
def effective_saturation(theta: float, theta_r: float, theta_s: float) -> float:
    """
    Compute effective saturation Se.

    Args:
        theta (float|array): volumetric soil moisture (cm3/cm3)
        theta_r (float): residual volumetric soil moisture (cm3/cm3)
        theta_s (float): saturated volumetric soil moisture (cm3/cm3)

    Returns:
        Se (float|array): effective saturation (-)
    """
    Se = (theta - theta_r) / (theta_s - theta_r)
    Se = np.clip(Se, 0, 1)
    return Se


def water_retention(psi: float, theta_r: float, theta_s: float, alpha: float,
                    n: float) -> float:
    """
    Compute soil moisture using van Genuchten's water retention equation.

    Args:
        psi (float|array): soil matric potential (cm)
        theta_r (float): residual volumetric soil moisture (cm3/cm3)
        theta_s (float): saturated volumetric soil moisture (cm3/cm3)
        alpha (float): inverse of the air entry suction (1/cm)
        n (float): pore-size distribution index (-)

    Returns:
        theta (float|array): volumetric soil moisture (cm3/cm3)
    """
    m = 1 - 1/n
    Se = (1 + (alpha * np.abs(psi))**n)**(-m)
    theta = theta_r + Se * (theta_s - theta_r)
    return theta


def mualem_conductivity(Se: float, Ks: float, n: float,
                        tortuosity: float = 0.5) -> float:
    """
    Compute unsaturated hydraulic conductivity using Mualem's model.

    Args:
        Se (float|array): effective saturation (-)
        Ks (float): saturated hydraulic conductivity (cm/h)
        n (float): pore-size distribution index (-)

    Returns:
        K (float|array): unsaturated hydraulic conductivity (cm/h)
    """
    m = 1 - 1/n
    K = Ks * Se**tortuosity * (1 - (1 - Se**(1/m))**m)**2
    return K

# ----------------------------- Horton equations ----------------------------- #


@np.vectorize
def Horton_Abstractions(pr: float, duration: float, f0: float, fc: float,
                        k: float) -> float:
    """
    Compute infiltration rate using Horton's equation.
    Based on soil classification, the common parameters used in the SWMM
    model include:

    | **SCS Soil Group** | **f₀ (mm/h)** | **fc (mm/h)** | **k (1/h)** |
    |--------------------|---------------|---------------|-------------|
    | A                  | 250           | 25.4          | 2           |
    | B                  | 200           | 12.7          | 2           |
    | C                  | 125           | 6.3           | 2           |
    | D                  | 76            | 2.5           | 2           |

    Args:
        pr (float|array): precipitation rate (mm/h)
        duration (float): Time duration of rainfall event (h)
        f0 (float): Dry or initial soil hydraulic conductivity (mm/h)
        fc (float): Saturated soil hydraulic conductivity (mm/h)
        k (float): Horton's method decay coefficient (1/h)

    Returns:
        f (float): Infiltration rate (mm/h)
    """
    f = fc + (f0 - fc) * np.exp(- k * duration)
    if pr <= f:
        f = pr
    return f


@np.vectorize
def Horton_EffectiveRainfall(pr: float, duration: float, f0: float, fc: float,
                             k: float) -> float:
    """
    Effective precipitation/runoff computation using Horton's model for 
    infiltration/losses.
    Based on soil classification, the common parameters used in the SWMM
    model include:

    | **SCS Soil Group** | **f₀ (mm/h)** | **fc (mm/h)** | **k (1/h)** |
    |--------------------|---------------|---------------|-------------|
    | A                  | 250           | 25.4          | 2           |
    | B                  | 200           | 12.7          | 2           |
    | C                  | 125           | 6.3           | 2           |
    | D                  | 76            | 2.5           | 2           |

    Args:
        pr (float|array): precipitation rate (mm/h)
        duration (float): Time duration of rainfall event (h)
        f0 (float): Dry or initial soil hydraulic conductivity (mm/h)
        fc (float): Saturated soil hydraulic conductivity (mm/h)
        k (float): Horton's method decay coefficient (1/h)

    Returns:
        pr_eff (float|array): Effective precipitation rate (mm/h)
    """
    F = Horton_Abstractions(pr, duration, f0, fc, k)
    pr_eff = pr - F
    return pr_eff

# ----------------------------- Philip's Equation ---------------------------- #


@np.vectorize
def Philip_Abstractions(pr: float, duration: float, S: float, K: float
                        ) -> float:
    """
    Compute infiltration rate using Philip's equation.

    Args:
        pr (float|array): precipitation rate (mm/h)
        duration (float): Time duration of rainfall event (h)
        S (float): Sorptivity coefficient (mm / h ^ 0.5)
        K (float): Saturated soil hydraulic conductivity (mm/h)

    Returns:
        f (float): Infiltration rate (mm/h)
    """
    if duration == 0:
        f = K
    else:
        f = 0.5 * S * (duration ** (-1/2)) + K
    if pr <= f:
        f = pr
    return f


@np.vectorize
def Philip_EffectiveRainfall(pr: float, duration: float, S: float, K: float
                             ) -> float:
    """
    Effective precipitation/runoff computation using Philip's model for 
    infiltration/losses.

    Args:
        pr (float|array): precipitation rate (mm/h)
        duration (float): Time duration of rainfall event (h)
        S (float): Adsorption coefficient (mm/h)
        K (float): Saturated soil hydraulic conductivity (mm/h)

    Returns:
        pr_eff (float|array): Effective precipitation rate (mm/h)
    """
    F = Philip_Abstractions(pr, duration, S, K)
    pr_eff = pr - F
    return pr_eff


# -------------------------- Green & Ampt equations -------------------------- #


@np.vectorize
def GreenAmpt_Abstractions(pr: float, duration: float, K: float, p: float,
                           theta: float, psi: float, h0: float = 10
                           ) -> float:
    """
    Compute infiltration rate using Green & Ampt's equation. The equation 
    to solve is the following implicit equation:

    F (t) = K*t + (p-theta)*(h0+psi)*ln (1+F/(p-theta)/(h0 + psi))

    which is solved using the Newton-Raphson method.

    Args:
        pr (float|array): precipitation rate (mm/h)
        duration (float): Time duration of rainfall event (h)
        K (float): Saturated soil hydraulic conductivity (mm/h)
        p (float): Soil porosity (-)
        theta (float): Soil fractional moisture (-)
        psi (float): Soil suction (mm). Highly dependant of soil moisture.
        h0 (float): water depth above the soil column (mm). Default to 10 mm. 


    Returns:
        f (float): Infiltration rate (mm/h)
    """
    if theta > p:
        text = f'theta: {theta} > porosity: {p}. '
        text += 'Soil cant have more moisture than the aviable void space!'
        raise ValueError(text)

    c1 = (p - theta)
    c2 = (h0 + psi)

    def _rootfunc(F):
        """
        Root finding function defined to solve with newton-raphson method.
        Right side minus left side of the Green & Ampt iterative function. 
        """
        return K * duration + c1 * c2 * np.log(1 + F/c1/c2) - F

    def _rootfunc_diff(F):
        """
        Derivative of the root finding function.
        """
        return c1*c2/(c1*c2 + F) - 1

    # Solve with Green & Ampt equation using Newton-Raphson method.
    if duration == 0 or (p-theta) == 0:
        f = K
    else:
        F = root_scalar(_rootfunc, x0=K*duration, fprime=_rootfunc_diff,
                        method='newton')
        f = K * (1 + c1 * c2 / F.root)
    if pr <= f:
        f = pr
    return f


@np.vectorize
def GreenAmpt_EffectiveRainfall(pr: float, duration: float, K: float, p: float,
                                theta: float, psi: float, h0: float = 10
                                ) -> float:
    """
    Effective precipitation/runoff computation using GreenAmpt's model for 
    infiltration/losses.

    Args:
        pr (float|array): precipitation rate (mm/h)
        duration (float): Time duration of rainfall event (h)
        K (float): Saturated soil hydraulic conductivity (mm/h)
        p (float): Soil porosity (-)
        theta (float): Soil fractional moisture (-)
        psi (float): Soil suction (mm). Highly dependant of soil moisture.
        h0 (float): water depth above the soil column (mm). Default to 10 mm. 

    Returns:
        pr_eff (float|array): Effective precipitation rate (mm/h)
    """
    F = GreenAmpt_Abstractions(pr, duration, K, p, theta, psi, h0)
    pr_eff = pr - F
    return pr_eff

# ------------------------ SCS curve number equations ------------------------ #


def cn_correction(cn_II: int | float | ArrayLike,
                  amc: str) -> float | ArrayLike:
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


def SCS_MaximumRetention(cn: int | float | ArrayLike,
                         cfactor: float = 25.4) -> float | ArrayLike:
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
    return cfactor*S if cn > 0 else np.nan


def SCS_EquivalentCurveNumber(pr: int | float | ArrayLike,
                              pr_eff: int | float | ArrayLike,
                              r: float = 0.2,
                              cfactor: float = 25.4
                              ) -> float | ArrayLike:
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


@np.vectorize
def SCS_EffectiveRainfall(pr: int | float,
                          cn: int | float,
                          r: float = 0.2,
                          **kwargs: float) -> float:
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
    if np.isnan(pr) or np.isnan(cn):
        return np.nan
    else:
        if pr < 0.:
            raise ValueError("Precipitation must be positive")
        if not 0. <= cn <= 100.:
            raise ValueError("CN must be between 0 and 100")
        if r <= 0.:
            raise ValueError("Initial abstraction ratio must be positive")
        S = SCS_MaximumRetention(cn, **kwargs)
        Ia = r * S
        if pr <= Ia:
            return 0.0
        return (pr - Ia) ** 2 / (pr - Ia + S)


@np.vectorize
def SCS_Abstractions(pr: int | float | ArrayLike,
                     cn: int | float | ArrayLike,
                     r: float = 0.2,
                     **kwargs: float) -> float | ArrayLike:
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
