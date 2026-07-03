'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 1969-12-31 21:00:00
 Modified by: Lucas Glasner,
 Modified time: 2024-05-06 16:40:13
 Description:
 Dependencies:
'''

import numpy as np
import pandas as pd
import xarray as xr
import warnings

from typing import Any, Tuple
from dataclasses import dataclass, fields as dc_fields
from numpy.typing import ArrayLike
from scipy.optimize import root_scalar
from .misc import raster_counts


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


@dataclass
class HortonParams:
    """Parameters for Horton infiltration model."""
    # Initial infiltration rate (mm/h)
    f0: float | ArrayLike | xr.DataArray
    # Final/saturated infiltration rate (mm/h)
    fc: float | ArrayLike | xr.DataArray
    # Decay coefficient (1/h)
    k: float | ArrayLike | xr.DataArray


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
        k (float): Horton's decay coefficient (1/h)

    Returns:
        f (float): Infiltration rate (mm/h)
    """
    f = fc + (f0 - fc) * np.exp(-k * duration)
    f = np.where(pr <= f, pr, f)
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


@dataclass
class PhilipParams:
    """Parameters for Philip infiltration model."""
    # Sorptivity coefficient (mm/h^0.5)
    S: float | ArrayLike | xr.DataArray
    # Saturated hydraulic conductivity (mm/h)
    K: float | ArrayLike | xr.DataArray


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
@dataclass
class GreenAmptParams:
    """Parameters for Green-Ampt infiltration model."""
    # Saturated hydraulic conductivity (mm/h)
    K: float | ArrayLike | xr.DataArray
    # Soil porosity (-)
    p: float | ArrayLike | xr.DataArray
    # Soil fractional moisture (-)
    theta_s: float | ArrayLike | xr.DataArray
    # Soil suction (mm)
    psi: float | ArrayLike | xr.DataArray
    # Water depth above soil column (mm)
    h0: float | ArrayLike | xr.DataArray = 10.0


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


@dataclass
class SCSParams:
    """Parameters for SCS Curve Number infiltration model."""
    cn: float | ArrayLike | xr.DataArray        # Curve Number (-)
    r: float | ArrayLike | xr.DataArray = 0.2   # Initial abstraction ratio (-)


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
    cn = np.asarray(cn, dtype=float)
    S = np.where(cn > 0, cfactor * (1000.0 / cn - 10.0), np.nan)
    return float(S) if S.ndim == 0 else S


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


def SCS_EffectiveRainfall(pr: int | float | ArrayLike,
                          cn: int | float | ArrayLike,
                          r: float = 0.2,
                          **kwargs: float) -> float | ArrayLike:
    """
    SCS formula for effective precipitation/runoff.

    Accepts scalars, lists, or numpy/xarray arrays. All operations are
    native numpy so no Python-level element iteration occurs.

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
    pr = np.asarray(pr, dtype=float)
    cn = np.asarray(cn, dtype=float)
    valid_cn = cn[~np.isnan(cn)]
    if np.any(valid_cn < 0) or np.any(valid_cn > 100):
        raise ValueError("CN must be between 0 and 100")
    S = SCS_MaximumRetention(cn, **kwargs)
    Ia = r * S
    nan_mask = np.isnan(pr) | np.isnan(cn)
    excess = pr - Ia
    pr_eff = np.where(nan_mask | (excess <= 0), 0.0, excess**2 / (excess + S))
    pr_eff = np.where(nan_mask, np.nan, pr_eff)
    return float(pr_eff) if pr_eff.ndim == 0 else pr_eff


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
        r (float, optional): Initial abstraction ratio. Default to 0.2.
        **kwargs are passed to SCS_MaximumRetention function


    Returns:
        (array_like): Losses/Abstraction/Infiltration
    """
    pr = np.asarray(pr, dtype=float)
    pr_eff = SCS_EffectiveRainfall(pr, cn, r=r, **kwargs)
    losses = pr - pr_eff
    return float(losses) if losses.ndim == 0 else losses

# ---------------------------------------------------------------------------- #


class InfiltrationModel:
    """
    Surface abstraction model. Computes infiltration losses and effective
    precipitation for a given rainfall rate time series using one of the
    supported infiltration methods.

    Supported methods: SCS, Horton, Philip, GreenAmpt.

    Attributes:
        method (str): Name of the infiltration method in use.
        params (dict): Active model parameters.
        infr (xr.DataArray | None): Computed infiltration rate [mm/hr].
            Set after calling infiltrate().
        pr_eff (xr.DataArray | None): Computed effective precipitation rate
            [mm/hr]. Set after calling infiltrate().
    """

    _REQUIRED_PARAMS: dict[str, set[str]] = {
        'SCS':       {'cn'},
        'Horton':    {'f0', 'fc', 'k'},
        'Philip':    {'S', 'K'},
        'GreenAmpt': {'K', 'p', 'theta_s', 'psi'},
    }

    _PARAMS_CLASS: dict[str, type] = {
        'SCS':       SCSParams,
        'Horton':    HortonParams,
        'Philip':    PhilipParams,
        'GreenAmpt': GreenAmptParams,
    }

    def __init__(self, method: str, **params: Any) -> None:
        """
        Initialize the InfiltrationModel with a chosen method and its
        corresponding parameters.

        Args:
            method (str): Infiltration method to use.
                Options: 'SCS', 'Horton', 'Philip', 'GreenAmpt'.
            **params: Method-specific parameters. Required parameters per
                method:

                - SCS: cn
                - Horton: f0, fc, k
                - Philip: S, K
                - GreenAmpt: K, p, theta_s, psi

        Raises:
            ValueError: If method is unknown or required parameters are
                missing.
        """
        required = self._REQUIRED_PARAMS.get(method)
        if required is None:
            valid = list(self._REQUIRED_PARAMS)
            raise ValueError(
                f"Unknown method '{method}'. Valid options: {valid}"
            )
        missing = required - params.keys()
        if missing:
            raise ValueError(
                f"Method '{method}' is missing required parameter(s): {missing}"
            )
        self.method = method
        self.params = params
        self.infr = None
        self.pr_eff = None

    def _infiltrate_SCS(self, pr: xr.DataArray, params: SCSParams,
                        **ufunc_kwargs: Any) -> xr.DataArray:
        """
        Compute infiltration using the SCS Curve Number method.

        Converts the precipitation rate series to cumulative depth, applies
        the SCS formula on the cumulative series using native xarray/numpy
        broadcasting, then differentiates back to an instantaneous
        infiltration rate.

        The SCS formula is pure arithmetic so apply_ufunc(vectorize=True) is
        avoided entirely. That approach would dispatch a Python call per
        spatial pixel (e.g. 10 000 calls for a 100×100 grid), each of which
        would in turn iterate element-wise through the time axis via
        @np.vectorize, resulting in millions of Python-level scalar ops.
        Direct broadcasting reduces this to a handful of C-level array ops.

        Args:
            pr (xr.DataArray): Precipitation rate [mm/hr] with a 'time'
                dimension.
            params (SCSParams): SCS model parameters (cn, r).
            **ufunc_kwargs: Accepted for API compatibility; not used.

        Returns:
            xr.DataArray: Infiltration rate [mm/hr] with the same shape
                as pr.
        """
        timestep = float(pr.time.values[1] - pr.time.values[0])
        pr_cum = pr.cumsum('time') * timestep

        # SCS formula - broadcasting handles (time, x, y) vs (x, y) natively
        S = 25.4 * (1000.0 / params.cn - 10.0)   # max retention [mm]
        Ia = params.r * S                          # initial abstraction [mm]
        excess = pr_cum - Ia
        pr_eff_cum = xr.where(excess <= 0, 0.0, excess**2 / (excess + S))
        # Put time first regardless of whether pr was 1-D or N-D and cn added
        # extra spatial dimensions via broadcasting.
        dims_order = ['time'] + [d for d in (pr_cum - pr_eff_cum).dims
                                 if d != 'time']
        infr_cum = (pr_cum - pr_eff_cum).transpose(*dims_order)

        infr = infr_cum.diff('time')
        infr = infr.reindex({'time': pr.time.values}) / timestep
        infr[0] = infr_cum.isel(time=0)
        return infr

    def _infiltrate_Horton(self, pr: xr.DataArray, params: HortonParams,
                           **ufunc_kwargs: Any) -> xr.DataArray:
        """
        Compute infiltration using Horton's exponential decay equation.

        Args:
            pr (xr.DataArray): Precipitation rate [mm/hr] with a 'time'
                dimension.
            params (HortonParams): Horton model parameters (f0, fc, k).
            **ufunc_kwargs: Accepted for API compatibility; not used.

        Returns:
            xr.DataArray: Infiltration rate [mm/hr] with the same shape
                as pr.
        """
        t = xr.DataArray(pr.time.values, dims=['time'])
        # Horton formula is pure arithmetic - broadcast over all dims natively
        f = params.fc + (params.f0 - params.fc) * np.exp(-params.k * t)
        infr = xr.where(pr <= f, pr, f)
        dims_order = ['time'] + [d for d in infr.dims if d != 'time']
        return infr.transpose(*dims_order)

    def _infiltrate_Philip(self, pr: xr.DataArray, params: PhilipParams,
                           **ufunc_kwargs: Any) -> xr.DataArray:
        """
        Compute infiltration using Philip's two-term equation.

        Args:
            pr (xr.DataArray): Precipitation rate [mm/hr] with a 'time'
                dimension.
            params (PhilipParams): Philip model parameters (S, K).
            **ufunc_kwargs: Accepted for API compatibility; not used.

        Returns:
            xr.DataArray: Infiltration rate [mm/hr] with the same shape
                as pr.
        """
        t = xr.DataArray(pr.time.values, dims=['time'])
        # Sentinel avoids t^(-0.5) = inf at t=0; masked by xr.where below
        t_safe = xr.where(t > 0, t, 1.0)
        f_raw = 0.5 * params.S * t_safe**(-0.5) + params.K
        f = xr.where(t <= 0, params.K, f_raw)
        infr = xr.where(pr <= f, pr, f)
        dims_order = ['time'] + [d for d in infr.dims if d != 'time']
        return infr.transpose(*dims_order)

    def _infiltrate_GreenAmpt(self, pr: xr.DataArray, params: GreenAmptParams,
                              **ufunc_kwargs: Any) -> xr.DataArray:
        """
        Compute infiltration using the Green-Ampt piston-flow equation.
        Solves the implicit Green-Ampt equation iteratively via
        Newton-Raphson at each time step.

        Args:
            pr (xr.DataArray): Precipitation rate [mm/hr] with a 'time'
                dimension.
            params (GreenAmptParams): Green-Ampt model parameters
                (K, p, theta_s, psi, h0).
            **ufunc_kwargs: Accepted for API compatibility; not used.

        Returns:
            xr.DataArray: Infiltration rate [mm/hr] with the same shape
                as pr.
        """
        t = xr.DataArray(pr.time.values, dims=['time'])
        c1  = params.p - params.theta_s      # available pore space
        c2  = params.h0 + params.psi         # effective suction head
        c12 = c1 * c2

        # Sentinels prevent divide-by-zero inside NR; final formula is still
        # correct because it multiplies by the original c12 (not c12_safe).
        # When K=0 the formula gives f=0 regardless of F, so F=1 is safe.
        t_safe   = xr.where(t   >  0,    t,    1.0)
        c12_safe = xr.where(c12 != 0,  c12,    1.0)
        F        = xr.where(params.K > 0, params.K * t_safe, 1.0)

        # Vectorised Newton-Raphson for F(t) = K*t + c12*ln(1 + F/c12)
        # Converges quadratically; 20 iterations is far more than sufficient
        for _ in range(20):
            fval  = params.K * t_safe + c12_safe * np.log(1.0 + F / c12_safe) - F
            dfval = c12_safe / (c12_safe + F) - 1.0
            F     = F - fval / dfval

        # Infiltration rate from cumulative F; c12=0 → f=K; K=0 → f=0
        f_raw = params.K * (1.0 + c12 / F)
        f = xr.where(t <= 0, params.K, f_raw)
        infr = xr.where(pr <= f, pr, f)
        dims_order = ['time'] + [d for d in infr.dims if d != 'time']
        return infr.transpose(*dims_order)

    def infiltrate(self, pr: xr.DataArray, **kwargs: Any):
        """
        Compute infiltration losses and effective precipitation for a
        given rainfall rate time series.

        Dispatches to the appropriate private method based on self.method,
        then stores the results in self.infr and self.pr_eff.

        Args:
            pr (xr.DataArray): Precipitation rate [mm/hr]. Must contain a
                'time' dimension whose coordinate values represent elapsed
                time in hours.
            **kwargs: Override or supplement model parameters for this call.
                Keys matching the active method's parameter fields (e.g. 'cn'
                for SCS) override the stored params. Any remaining keys are
                forwarded to xr.apply_ufunc (e.g. dask='parallelized').

        Raises:
            ValueError: If pr does not contain a 'time' dimension.

        Returns:
            self: Updated instance with infr and pr_eff set.
        """
        if 'time' not in pr.dims:
            raise ValueError("Precipitation array must have 'time' dimension")

        all_params = {**self.params, **kwargs}

        ParamsClass = self._PARAMS_CLASS[self.method]
        field_names = {f.name for f in dc_fields(ParamsClass)}
        model_params = ParamsClass(**{k: v for k, v in all_params.items()
                                      if k in field_names})
        ufunc_kwargs = {k: v for k, v in all_params.items()
                        if k not in field_names}

        _dispatch = {
            'SCS':       self._infiltrate_SCS,
            'Horton':    self._infiltrate_Horton,
            'Philip':    self._infiltrate_Philip,
            'GreenAmpt': self._infiltrate_GreenAmpt,
        }
        infr = _dispatch[self.method](pr, model_params, **ufunc_kwargs)

        pr_eff = pr - infr
        pr_eff = pr_eff.where(pr_eff >= 0).fillna(0)

        infr.attrs = {'standard_name': 'infiltration rate', 'units': 'mm/hr'}
        pr_eff.attrs = {'standard_name': 'effective precipitation rate',
                        'units': 'mm/hr'}

        pr_eff.name = 'pr_eff'
        infr.name = 'infr'

        self.params = all_params
        self.infr = infr
        self.pr_eff = pr_eff

# ---------------------------------------------------------------------------- #


class LandSurface:
    """
    A class for processing and analyzing Land Surface datasets for hydrological
    analysis.
    """

    def __init__(self, lulc: xr.Dataset | xr.DataArray):
        """
        Initializes the LandSurface class with a given land dataset.

        Args:
            lulc (xr.DataArray | xr.Dataset):
                A 2D xarray.DataArray or xr.Dataset representing the digital
                surface land cover properties. Might be a single 2D raster or
                multiple rasters in a dataset.
        """
        if isinstance(lulc, xr.DataArray):
            lulc = lulc.to_dataset(name=lulc.name)
        for var in lulc.variables:
            lulc[var] = lulc[var].squeeze().copy()
            # lulc[var] = lulc[var].where(lulc[var] != -9999)
        self.lulc = lulc

    def _process_lulc(self, **kwargs):
        """
        Process the land use/land cover (LULC) data to compute area
        distributions and other relevant statistics.
        """
        if 'cn' in self.lulc.variables:
            if 'cn1' not in self.lulc.variables:
                self.lulc['cn1'] = cn_correction(self.lulc['cn'], amc='I')
            if 'cn2' not in self.lulc.variables:
                self.lulc['cn2'] = cn_correction(self.lulc['cn'], amc='II')
            if 'cn3' not in self.lulc.variables:
                self.lulc['cn3'] = cn_correction(self.lulc['cn'], amc='III')
            if 'amc' in kwargs.keys():
                self.lulc['cn'] = cn_correction(self.lulc['cn'],
                                                amc=kwargs['amc'])

        # LULC derived params
        counts, averages = [], []
        for var in self.lulc.data_vars:
            var = self.lulc[var]
            counts.append(raster_counts(var, output_type=1))
            try:
                averages.append(var.mean().item())
            except Exception as e:
                averages.append(np.nan)
                warnings.warn(f'Runtime Exception: {e}')
        self.lulc_counts = pd.concat(counts, keys=self.lulc.data_vars)
        self.lulc_counts = self.lulc_counts.stack().unstack(1).T
        self.lulc_params = pd.Series(averages, index=self.lulc.data_vars)

    def get_equivalent_curvenumber(self,
                                   pr_range: Tuple[float, float] = (1., 1000.),
                                   **kwargs: Any) -> pd.Series:
        """
        Calculate the dependence of the watershed curve number on precipitation
        due to land cover heterogeneities.

        This routine computes the equivalent curve number for a heterogeneous
        basin as a function of precipitation. It takes into account the
        distribution of curve numbers within the basin and the corresponding
        effective rainfall for a range of precipitation values.

        Args:
            pr_range (tuple): Minimum and maximum possible precipitation (mm).
            **kwargs: Additional keyword arguments to pass to the
                SCS_EffectiveRainfall and SCS_EquivalentCurveNumber routine.

        Returns:
            pd.Series: A pandas Series representing the equivalent curve number
                as a function of precipitation, where the index corresponds to
                precipitation values and the values represent the equivalent
                curve number.
        """
        if 'cn' not in self.lulc.variables:
            raise ValueError(
                "Curve number ('cn') variable not found in LULC dataset.")
        # Precipitation range
        pr = np.linspace(pr_range[0], pr_range[1], 1000)
        pr = np.expand_dims(pr, axis=-1)

        # Curve number counts
        cn_counts = raster_counts(self.lulc.cn)
        weights, cn_values = cn_counts['counts'].values, cn_counts['cn'].values
        cn_values = np.expand_dims(cn_values, axis=-1)

        # Broadcast curve number and pr arrays
        broad = np.broadcast_arrays(cn_values, pr.T)

        # Get effective precipitation
        pr_eff = SCS_EffectiveRainfall(pr=broad[1], cn=broad[0], **kwargs)
        pr_eff = (pr_eff.T * weights).sum(axis=-1)

        # Compute equivalent curve number for hetergeneous basin
        curve = SCS_EquivalentCurveNumber(pr[:, 0], pr_eff, **kwargs)
        curve = pd.Series(curve, index=pr[:, 0])
        curve = curve.sort_index()
        return curve

    def infiltrate(self, pr: xr.DataArray, method: str,
                   **kwargs: Any) -> Tuple[xr.DataArray, xr.DataArray]:
        """
        Compute infiltration losses and effective precipitation for a given
        rainfall rate time series.

        Args:
            pr (xr.DataArray): Precipitation rate [mm/hr] with a 'time'
                dimension.
            method (str): Infiltration method to use.
                Options: 'SCS', 'Horton', 'Philip', 'GreenAmpt'.
            **kwargs: Method-specific parameter overrides (e.g. cn=75 for
                SCS). Any parameter not recognised by the method is forwarded
                to xr.apply_ufunc inside InfiltrationModel.

        Raises:
            ValueError: If required parameters for the chosen method are
                missing from both lulc_params and kwargs.

        Returns:
            Tuple[xr.DataArray, xr.DataArray]: Tuple containing effective
                precipitation and infiltration losses.
        """
        # Pull matching params from lulc_params as defaults
        lulc_based = {}
        if hasattr(self, 'lulc_params'):
            required = InfiltrationModel._REQUIRED_PARAMS.get(method, set())
            for p in required:
                if p in self.lulc_params.index and p not in kwargs:
                    lulc_based[p] = self.lulc_params[p]

        model = InfiltrationModel(method=method, **lulc_based, **kwargs)
        model.infiltrate(pr)
        return model.pr_eff, model.infr
