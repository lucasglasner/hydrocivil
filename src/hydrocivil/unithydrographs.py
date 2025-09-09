'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 1969-12-31 21:00:00
 Modified by: Lucas Glasner,
 Modified time: 2024-05-06 16:40:29
 Description:
 Dependencies:
'''

import warnings
import numpy as np
import pandas as pd
import scipy.signal as sg
import geopandas as gpd
import matplotlib.pyplot as plt

from math import gamma
from typing import Any, Tuple, Type
from scipy.interpolate import interp1d
from shapely.geometry import Point

from .geomorphology import concentration_time
from .global_vars import CHILE_UH_LINSLEYPARAMS, CHILE_UH_GRAYPARAMS
from .global_vars import CHILE_UH_LINSLEYPOLYGONS, CHILE_UH_GRAYPOLYGONS


# ----------------------------- UNIT HYDROGRAPHS ----------------------------- #

def SUH_Clark(area: float, tc: float, tstep: float,
              R: float = None, X: float = None,
              timearea: pd.Series = None, tolerance: float = 1e-5,
              interp_kwargs: dict = {'kind': 'quadratic'}
              ) -> Tuple[pd.Series, pd.Series]:
    """
    Clark (1945) instantaneous unit hydrograph model. 

    The Clark's model indicates that the basin response can be computed from a
    time-area relationship, the concentration time and a storage coefficient R.
    The model assumes that the basin behaves like a linear reservoir following
    the equation:

        dS/dt = I(t) - Q(t)
        S(t) = R * Q(t)

    If a 1mm pulse is homogeneously and instantaneously distributed on the
    basin area the input runoff of Clark linear model will be exactly equal
    to the time-area curve. So, solving the previous equation with a finite
    difference approach leads to: 

        c = (dt / (R + 0.5 * dt))
        Q(t)= c * I(t) + (1 - c) * Q(t-1) 

    which can be solved iteratively from a given time-area curve I(t), 
    a timestep dt and the R storage parameter.

    The time-area curve of the basin is a complex function of the basin 
    geomorphology and hydraulic properties, and should go between 0 in the 
    basin outlet to the concentration time on the farthest hydrological point. 

    The R parameter is a function of the basin properties, however is 
    commonly estimated through empirical formulas. The most common is to 
    compute it from the concentration time with something like:

        X = R / (tc + R)
        R = X * t_c / (1 - X)

    where X is a coefficient determined through regional analyses. From HEC-HMS
    docs: "Smaller values of X result in short, steeply rising unit hydrographs
    and may be representative of urban watersheds. Larger values of X result
    in broad, slowly rising unit hydrographs and may be representative of flat,
    swampy watersheds."

    References:
        Clark, C. O. (1945). Storage and the unit hydrograph. Transactions of
        the American Society of Civil Engineers, 110(1), 1419-1446.

    Args:
        area (float): Basin area (km2)
        tc (float): Basin concentration time (hours)
        R (float): Linear reservoir storage coefficient (hours). Usually an 
            empirical function of the concentration time.
        X (float): Fractional coefficient used to derive R from tc with the
            formula X = R / (tc + R).
        tstep (float): Unit hydrograph discretization time step in hours.
        timearea (pandas.Series, optional): A user defined dimensionless
            time-area curve with dimensionless time (t/tc) in the index and 
            dimensionless area in values (A/A_basin). If None will use the 
            HEC-HMS default time-area curve. Defaults to None.
        tolerance (float, optional): Tolerance for solving the linear
            reservoir ODE. Defaults to 1e-5.
        interp_kwargs (dict, optional): args passed to
            scipy.interpolation.interp1d function. 
            Defaults to {'kind':'quadratic'}.

    Returns:
        uh, (qp, tp, tc, R) (tuple)
    """
    text = "SUH_Clark() missing 1 required positional argument: 'R' or 'X'"
    if R is None:
        if X is not None:
            R = X * tc / (1 - X)
        else:
            raise TypeError(text)

    c = 2*tstep/(2*R+tstep)
    if timearea is None:
        t_shape = np.arange(0, 1+0.1, 0.1)
        At_shape = np.full(t_shape.shape, np.nan)
        At_shape[t_shape <= 1/2] = 1.414*(t_shape[t_shape <= 1/2])**1.5
        At_shape[t_shape > 1/2] = 1-1.414*(1-t_shape[t_shape > 1/2])**1.5
        timearea = pd.Series(At_shape, index=t_shape)
    timearea.index = timearea.index*tc

    # Interpolate to new time resolution
    ntime = np.arange(timearea.index[0], timearea.index[-1]+tstep, tstep)
    f = interp1d(timearea.index, timearea.values, fill_value='extrapolate',
                 **interp_kwargs)
    timearea = pd.Series(f(ntime), index=ntime)
    timearea = timearea.diff().fillna(0)
    q_in = timearea*1e3/3600

    # Main loop
    uh, time = [0], [0]
    i = 0
    while True:
        i = i+1
        if i < len(q_in):
            qout = q_in.iloc[i]*c+uh[i-1]*(1-c)
        else:
            qout = uh[i-1]*(1-c)
        uh.append(qout)
        time.append(tstep*i)
        if uh[i] < tolerance:
            break
    uh = pd.Series(uh, index=time)
    uh = (uh.shift(1)+uh)/2
    uh = uh.where(uh > 0).fillna(0)
    uh = uh*area

    # Ensure that the unit hydrograph acummulates a volume of 1mm
    volume = np.trapezoid(uh, uh.index*3600)/1e6/area*1e3
    uh = uh/volume
    params = (uh.max(), uh.idxmax(), tc, R)
    params = pd.Series(params, index=['qpeak', 'tpeak', 'tc', 'R'])
    uh.name, params.name = 'Clark_m3 s-1 mm-1', 'Params_Clark'
    return uh, params


def SUH_SCS(area: float, tc: float, tstep: float, prf: float = 484,
            threshold: float = 1e-10,
            interp_kwargs: dict = {'kind': 'quadratic'}
            ) -> Tuple[pd.Series, pd.Series]:
    """
    U.S.A Soil Conservation Service (SCS) synthetic unit hydrograph.

    The SCS model for unit hydrograph calculation is based on a triangular unit
    hydrograph and a dimensionless unit hydrograph. The model assumes that the
    unit hydrograph can be characterized as a function of the peak flow 'q_p'
    and peak time 't_p'. The equations are:

    t_p = 0.6 * t_c + dt / 2
    q_p = prf * area / t_p

    where t_c is the basin concentration time, dt the reference rain duration
    and timestep and prf the peak rate factor. The dimensionless unit
    hydrograph comes from solving the so called 'gamma equation', which is:

    q / q_p = exp(m) * ((t / t_p) ^ m) * exp(-m * (t / t_p))

    where 'm' is a shape parameter related to the prf by the equation:

    prf = C * m**(m + 1) / exp(m) / Γ(m+1)

    where Γ is the gamma function and C a constant equal to 645.33 for
    imperial units, and 0.2787 for SI units. The peak rate factor
    for the traditional SCS unit hydrograph is equal to PRF = 484 (imperial
    units) or PRF = 0.208 (SI units). However recent literature (?? HEC-HMS ??)
    says it can vary with basin geomorphological parameters for a range of at
    least 100 - 600 (imperial units).

    References:
        Bhunya, P. K., Panda, S. N., & Goel, M. K. (2011).
        Synthetic unit hydrograph methods: a critical review.
        The Open Hydrology Journal, 5(1).

        Natural Resources Conservation Service. Chapter 16 - Hydrographs.
        National Engineering Handbook Part 630 - Hydrology. United States
        Department of Agriculture (USDA), 2007

        Chow Ven, T., Te, C. V., RC, M. D., & Mays, L. W. (1988).
        Applied hydrology. McGraw-Hill Book Company.

    Args:
        area (float): Basin area (km2)
        tc (float): Basin concentration time (hours)
        tstep (float): Unit hydrograph discretization time step in hours.
        prf (float): peak rate factor (imperial units). Defaults to 484. 
        threshold (float): Iteration error for the m parameter calculation 
            based on the given peak rate factor (prf). Defaults to 1e-10. 
        interp_kwargs (dict, optional): args passed to
            scipy.interpolation.interp1d function. 
            Defaults to {'kind':'quadratic'}.

    Returns:
        uh, (qp, tp, tb, tstep) (tuple):
            (unit hydrograph),
            (Peak runoff (L/s/km2/mm), peak time (hours), base time (hours),
            time step (hours)))

    """
    def _iterfunc(target_prf: float, m: float) -> float:
        """
        Args:
            target_prf (float): target peak rate factor
            m (float): input m parameter

        Returns:
            (float): updated m parameter
        """
        C = 645.33
        a = target_prf * np.exp(m)
        b = gamma(m+1) / C
        return (a*b)**(1/(m+1))

    def _solve4m(target_prf: float, threshold: float) -> float:
        """
        This function computes m for a given prf.

        Args:
            target_prf (float): peak rate factor (imperial units)
            threshold (float): Iteration threshold.
        """
        m0 = 0
        while True:
            m = _iterfunc(target_prf, m0)
            if abs(m-m0) < threshold:
                return m
            else:
                m0 = m

    # Unit hydrograph shape
    m = _solve4m(prf, threshold=threshold)
    t_shape = [np.arange(0, 2+0.1, 0.1),
               np.arange(2.2, 4+0.2, 0.2),
               np.arange(4.5, 20+0.5, 0.5)]
    t_shape = np.hstack(t_shape)
    q_shape = np.exp(m)*t_shape**m*np.exp(-m*t_shape)

    # Unit hydrograph paremeters
    tp = tc*0.6+tstep/2
    tb = 2.67*tp
    qp = prf*area/tp*(0.0283/2.58/25.4)  # International units
    uh = pd.Series(qp*q_shape, index=t_shape*tp)

    # Interpolate to new time resolution
    ntime = np.arange(uh.index[0], uh.index[-1]+tstep, tstep)
    f = interp1d(uh.index, uh.values,
                 fill_value='extrapolate', **interp_kwargs)
    uh = f(ntime)
    uh = pd.Series(uh, index=ntime)
    uh = uh.where(uh > 0).fillna(0)

    # Ensure that the unit hydrograph acummulates a volume of 1mm
    volume = np.trapezoid(uh, uh.index*3600)/1e6/area*1e3
    uh = uh/volume
    params = (qp, tp, tb, tstep, prf)
    params = pd.Series(
        params, index=['prf', 'qpeak', 'tpeak', 'tbase', 'tstep'])
    uh.name, params.name = 'SCS_m3 s-1 mm-1', 'Params_SCS'
    return uh, params


def tstep_correction(tstep: float, tp: float) -> float:
    """
    This functions checks if the selected timestep can be used
    as the unit hydrograph time resolution (unitary time) for Snyder
    or Linsley methods.

    Args:
        tstep (float): Desired time step in hours
        tp (float): Raw unit hydrograph peak time (tu) in hours.

    Raises:
        RuntimeError: If selected tstep exceeds a 10% change of
            unit hydrograph storm duration/peak time.

    Returns:
        float: Fixed timestep and peak time in hours
    """
    tu = tp/5.5
    if tstep > tu*0.5:
        warnings.warn('tstep exceeds tu/2, changing tstep to tu = tp/5.5')
        return tu, tp
    if (tstep < tu+0.1) and (tstep > tu-0.1):
        return tstep, tp
    else:
        tp = tp+0.25*(tstep-tu)
        return tstep, tp


def SUH_Gray(area: float, mriverlen: float, meanslope: float,
             a: float, b: float,
             interp_kwargs: dict = {'kind': 'quadratic'}
             ) -> Tuple[pd.Series, pd.Series]:
    """
    Gray's' method assumes a SUH that follows the gamma function, which
    reflects the theoretical result of a basin made of an infinite series
    of lineal reservoirs.

    The model suggests that the dimensionless unit hydrograph can be obtained
    by the equation:

    q_shape = 25 * y^(y+1) * exp(-y * t_shape) * (t_shape)^y / gamma(y + 1)

    where q_shape = q / q_peak and t_shape = t / t_p are the dimensionless
    coordinates of the unit hydrograph. y is the model parameter and gamma
    is the gamma function.

    Based on some regression analysis Gray's method says that the y parameter
    is related to the peak time t_p by the equation:

        t_p / y = (2.676 / t_p + 0.0139)^(-1)
        y = 2.676 + 0.0139 * t_p

    where (t_p / y) would be a measure of the storage property of the watershed,
    or the travel time required for water to pass through a given reach.

    (t_p / y) is naturally dependant of the basin geomorphology so the method
    suggests a relationship of the kind:

        t_p / y = a * ( L / sqrt(S))^b

    where a and b are empirical parameters possibly dependant of geomorphology
    and land cover. Note that this equation in relation with the previous one
    allows to compute the y parameter frmo t_p, so the only real parameters
    of the model are the empirical a and b of the geomorphology equation.


    References:
        Bhunya, P. K., Panda, S. N., & Goel, M. K. (2011).
        Synthetic unit hydrograph methods: a critical review.
        The Open Hydrology Journal, 5(1).

        Bras, R. L. (1990). Hydrology: an introduction to hydrologic science.

    Args:
        area (float): Basin area (km2)
        mriverlen (float): Main channel length (km)
        meanslope (float): Basin mean slope (m/m)
        a, b (float): Gray's model parameters.
        tstep (float): Unit hydrograph target unitary time (tu) in hours.
        interp_kwargs (dict, optional): args passed to
            scipy.interpolation.interp1d function. 
            Defaults to {'kind':'quadratic'}.

    """
    alpha, beta = 2.676, 0.0139
    y = alpha/(1-beta*a*((mriverlen / np.sqrt(meanslope))**b))
    tp = (y-alpha)/beta  # minutes
    tp = tp/60  # hours

    tstep = tp/5.5

    t_shape = np.arange(0, 10+0.1, 0.1)
    q_shape = 25*y**(y+1)*np.exp(-y*t_shape)*(t_shape)**(y)/gamma(y+1)

    uh = pd.Series(q_shape, index=t_shape*tp)
    uh = uh*area/360/(0.25*tp)

    # Interpolate to new time resolution
    ntime = np.arange(uh.index[0], uh.index[-1]+tstep, tstep)
    f = interp1d(uh.index, uh.values, fill_value='extrapolate',
                 **interp_kwargs)
    uh = f(ntime)
    uh = pd.Series(uh, index=ntime)
    uh = uh.where(uh > 0).fillna(0)

    # Ensure that the unit hydrograph acummulates a volume of 1mm
    volume = np.trapezoid(uh, uh.index*3600)/1e6/area*1e3
    uh = uh/volume
    params = (uh.max(), tp, tstep, y, a, b)
    params = pd.Series(params, index=['qpeak', 'tpeak', 'tstep',
                                      'y', 'a', 'b'])
    uh.name, params.name = 'Gray_m3 s-1 mm-1', 'Params_Gray'
    return uh, params


def SUH_Linsley(area: float, mriverlen: float, out2centroidlen: float,
                meanslope: float,
                C_t: float, n_t: float,
                C_p: float, n_p: float,
                C_b: float, n_b: float,
                interp_kwargs: dict = {'kind': 'quadratic'}
                ) -> Tuple[pd.Series, pd.Series]:
    """
    Linsley unit hydrograph (UH) is a similar formulation of the well known
    Snyder UH. The difference is that Linsley's UH uses a different formula
    for the peak flow like this:

        tp = Ct * (L * Lg / sqrt(S)) ^ nt
        qp = Cp * tp ^ np
        tb = Cb * tp ^ nb

    Where L, Lg and S are basin main channel length, distance between the basin
    outlet and centroid, and basin mean slope.

    Ct, nt are shape parameters for the peak time,
    Cp, np are shape parameters for the peak flow,
    Cb, nb are shape parameters for the base flow

    *** Shape paremeters probably depend on hydrological soil properties,
    land cover use and geomorphology.

    Just like the Snyder UH, the Linsley UH is defined for a rain duration of

        tu = tp / 5.5

    References:
        Manual de calculo de crecidas y caudales minimos en cuencas sin
        informacion fluviometrica. Republica de Chile, Ministerio de Obras
        Publicas (MOP), Dirección General de Aguas (DGA) (1995).

        Metodo para la determinación de los hidrogramas sintéticos en Chile,
        Arteaga F., Benitez A., División de Estudios Hidrológicos,
        Empresa Nacional de Electricidad S.A (1985).

    Args:
        area (float): Basin area (km2)
        mriverlen (float): Main channel length (km)
        out2centroidlen (float): Distance from basin outlet to centroid (km)
        meanslope (float): Basin mean slope (m/m)
        C_t, n_t, C_p, n_p, C_b, n_b (float): Linsley model parameters.
        interp_kwargs (dict, optional): args passed to
            scipy.interpolation.interp1d function. 
            Defaults to {'kind':'quadratic'}.


    Returns:
        uh, (qpR, tpR, tbR, tuR) (tuple):
            (unit hydrograph),
            (Peak runoff (m3/s/km2/mm), peak time (hours), time step (hours)))
    """

    tp = C_t * (mriverlen*out2centroidlen / np.sqrt(meanslope))**n_t

    # Adjust storm duration to the UH timestep
    tpR = tp
    tstep = tp/5.5
    qpR = C_p*tpR**(n_p)
    tbR = C_b*tpR**(n_b)

    # Unit hydrograph shape
    t_shape = np.array([0, 0.3, 0.5, 0.6, 0.75, 1, 1.3, 1.5, 1.8, 2.3, 2.7, 3])
    q_shape = np.array([0, 0.2, 0.4, 0.6, 0.80, 1, 0.8, 0.6, 0.4, 0.2, 0.1, 0])

    # Compute unit hydrograph of duration equals tu
    uh = np.array(q_shape)*qpR
    uh = pd.Series(uh, index=np.array(t_shape)*tpR)

    # Interpolate to new time resolution
    ntime = np.arange(uh.index[0], uh.index[-1] + tstep, tstep)
    f = interp1d(uh.index, uh.values, fill_value='extrapolate',
                 **interp_kwargs)
    uh = f(ntime)
    uh = pd.Series(uh, index=ntime)
    uh = uh.where(uh > 0).fillna(0)

    # Ensure that the unit hydrograph acummulates a volume of 1mm
    volume = np.trapezoid(uh, uh.index*3600)/1e6  # mm
    uh = uh/volume
    params = (qpR/volume*area/1e3, tpR, tbR,
              tstep, C_t, n_t, C_p, n_p, C_b, n_b)
    params = pd.Series(params, index=['qpeak', 'tpeak', 'tbase', 'tstep',
                                      'C_t', 'n_t', 'C_p', 'n_p',
                                      'C_b', 'n_b'])
    uh.name, params.name = 'Linsley_m3 s-1 mm-1', 'Params_Linsley'
    return uh*area/1e3, params

# ------------------------------- MAIN CLASSES ------------------------------- #


class LumpedUnitHydrograph():
    """
    Class for building unit hydrographs of river basins based on
    geomorphometric and land use properties. Supports Clark, SCS, Linsley,
    and Gray methods. For Gray and Linsley, optional pre-calibrated Chilean
    parameters are available, following the national flood manual.
    """

    def __init__(self, method: str, geoparams: dict | pd.Series
                 ) -> Type['LumpedUnitHydrograph']:
        """
        Synthetic unit hydrograph (SUH) constructor.

        Args:
            method (str): Type of synthetic unit hydrograph to use.
                Options: 'SCS', Linsley, ...
            geoparams (dict): Input geomorphologic and land use parameters.
        """
        self.method = method
        self.geoparams = geoparams
        self.kwargs = {}
        self.timestep = None
        self.unithydro = None
        self.scurve = None
        self.params = None

    def _Clark(self, timestep: float, tc_formula: str = 'SCS', **kwargs: Any
               ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute the unit hydrograph following Clark's model.

        Args:
            timestep (float): timestep in hours. Note that this is the same
                as the unit hydrograph excess rain duration.
            tc_formula (str, optional): Empirical formula used for computing
                the time of concentration. Options: 'California', 'Giandotti',
                'Kirpich', 'SCS', 'Spain'. Defaults to 'SCS'.

        Returns:
            (tuple): tuple with the unit hydrograph time series and the
                respective table of parameters
        """
        area = self.geoparams['area']
        if 'tc' in self.geoparams.keys():
            tc = self.geoparams['tc']
            uh, uh_params = SUH_Clark(tstep=timestep, area=area, tc=tc)
        else:
            tc = concentration_time(method=tc_formula, **self.geoparams)/60
            uh, uh_params = SUH_Clark(tstep=timestep, area=area, tc=tc,
                                      **kwargs)
        return uh, uh_params

    def _SCS(self, timestep: float, tc_formula: str = 'SCS', **kwargs: Any
             ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute the unit hydrograph following the SCS model.

        Args:
            timestep (float): timestep in hours. Note that this is the same
                as the unit hydrograph excess rain duration.
            tc_formula (str, optional): Empirical formula used for computing
                the time of concentration. Options: 'California', 'Giandotti',
                'Kirpich', 'SCS', 'Spain'. Defaults to 'SCS'.
            **kwargs: Additional parameters for the SCS model. If tc given in
                      kwargs, it will be used instead of computing it from the
                      tc_formula.

        Returns:
            (tuple): tuple with the unit hydrograph time series and the
                respective table of parameters
        """
        area = self.geoparams['area']
        if 'tc' in self.geoparams.keys():
            tc = self.geoparams['tc']
            uh, uh_params = SUH_SCS(tstep=timestep, area=area, tc=tc)
        else:
            tc = concentration_time(method=tc_formula, **self.geoparams)/60
            uh, uh_params = SUH_SCS(tstep=timestep, area=area, tc=tc,
                                    **kwargs)
        return uh, uh_params

    def _Linsley(self, DGAChileParams: bool = False, DGAChileZone: bool = None,
                 **kwargs: Any) -> Tuple[pd.Series, pd.Series]:
        """
        Compute the unit hydrograph following the Linsley model.
        Args:
            DGAChileParams (bool, optional): Wheter to use pre-calibrated
                parameters for Chilean basins. Defaults to False.
            DGAChileZone (str, optional): Zone of the pre-calibrated parameters 
                for Chilean basins following Arteaga & Benitez and DGA methods.
                Options: 'I', 'II', 'III', 'IV'. If DGAChileZone == None the 
                function will ask you to give the Linsley model parameters.
                Defaults to None.

        Returns:
            (tuple): tuple with the unit hydrograph time series and the
                respective table of parameters
        """
        tparams = ['area', 'mriverlen', 'out2centroidlen', 'meanslope']
        if DGAChileParams:
            if DGAChileZone is None:
                epsg_code = f"EPSG:{self.geoparams['EPSG']}"
                x = self.geoparams['centroid_x']
                y = self.geoparams['centroid_y']
                centroid = gpd.GeoSeries(Point(x, y), crs=epsg_code)
                centroid = centroid.repeat(len(CHILE_UH_LINSLEYPOLYGONS))
                centroid = centroid.reset_index(drop=True).to_crs('EPSG:4326')
                mask = centroid.within(CHILE_UH_LINSLEYPOLYGONS.geometry)
                if mask.sum() == 0:
                    text = 'Basin is outside the geographical limits'
                    text += ' allowed by the Chilean Linsley method.'
                    raise RuntimeError(text)
                else:
                    DGAChileZone = CHILE_UH_LINSLEYPOLYGONS[mask]
                    DGAChileZone = DGAChileZone.zone.item()

            coefs = CHILE_UH_LINSLEYPARAMS[DGAChileZone]
            uh, uh_params = SUH_Linsley(**self.geoparams[tparams], **coefs,
                                        **kwargs)
        else:
            uh, uh_params = SUH_Linsley(**self.geoparams[tparams], **kwargs)
        return uh, uh_params

    def _Gray(self, DGAChileParams: bool = False, **kwargs: Any
              ) -> Tuple[pd.Series, pd.Series]:
        """
        Compute the unit hydrograph following the Gray model.
        Args:
            DGAChileParams (bool, optional): Wheter to use pre-calibrated
                parameters for Chilean basins. Defaults to False.

        Returns:
            (tuple): tuple with the unit hydrograph time series and the
                respective table of parameters
        """
        tparams = ['area', 'mriverlen', 'meanslope']
        if DGAChileParams:
            epsg_code = f'EPSG:{self.geoparams['EPSG']}'
            x = self.geoparams['centroid_x']
            y = self.geoparams['centroid_y']
            centroid = gpd.GeoSeries(Point(x, y), crs=epsg_code)
            centroid = centroid.repeat(len(CHILE_UH_GRAYPOLYGONS))
            centroid = centroid.reset_index(drop=True).to_crs('EPSG:4326')
            mask = centroid.within(CHILE_UH_GRAYPOLYGONS.geometry)
            if mask.sum() == 0:
                text = 'Basin is outside the geographical limits'
                text += ' allowed by the Chilean Gray method.'
                raise RuntimeError(text)

            a, b = CHILE_UH_GRAYPARAMS['a'], CHILE_UH_GRAYPARAMS['b']
            uh, uh_params = SUH_Gray(**self.geoparams[tparams], a=a, b=b,
                                     **kwargs)
        else:
            uh, uh_params = SUH_Gray(**self.geoparams[tparams], **kwargs)
        return uh, uh_params

    def get_SHydrograph(self) -> pd.Series:
        """
        This function computes the S-Curve or S-Hydrograph which is independent
        of the storm duration and can be used for computing the UH of a
        different excess-rain duration.

        Returns:
            (pandas.Series): S-Unit Hydrograph. 
        """
        uh = self.unithydro
        sums = [uh.shift(i) for i in range(len(uh)+1)]
        S_uh = pd.concat(sums, axis=1).sum(axis=1)
        return S_uh

    def update_duration(self, duration: float,
                        interp_kwargs: dict = {'kind': 'quadratic'}
                        ) -> Type['LumpedUnitHydrograph']:
        """
        This function uses the S-Curve to update the unit hydrograph
        duration and the respective parameters.

        Args:
            duration (float): New storm duration (equal to time resolution)
            interp_kwargs (dict, optional): args passed to
                scipy.interpolation.interp1d function. 
                Defaults to {'kind':'quadratic'}.

        Returns:
            LumpedUnitHydrograph: Updated instance of the class with the
                new duration.

        Args:
            duration (float): New storm duration (equal to time resolution)
            interp_kwargs (dict, optional): args passed to
                scipy.interpolation.interp1d function. 
                Defaults to {'kind':'quadratic'}.


        Returns:
            self: Updated Class
        """
        time, scurve = self.unithydro.index, self.scurve
        new_time = np.arange(time[0], time[-1]+duration, duration)
        interp_func = interp1d(time, scurve.values, fill_value='extrapolate',
                               **interp_kwargs)
        scurve_new = pd.Series(interp_func(new_time), index=new_time)
        uh_new = (scurve_new-scurve_new.shift(1).fillna(0))
        uh_new = uh_new.where(uh_new > 0).dropna()
        uh_new.loc[uh_new.index[-1]+duration] = 0
        uh_new.loc[0] = 0
        uh_new = uh_new.sort_index()

        # Ensure that the unit hydrograph acummulates a volume of 1mm
        volume = np.trapezoid(uh_new, uh_new.index*3600)
        volume = volume/self.geoparams['area']/1e3  # mm
        uh_new = uh_new/volume

        params_new = pd.Series([uh_new.max(), uh_new.idxmax(), duration],
                               index=['qpeak', 'tpeak', 'tstep'])

        # Update
        self.timestep = duration
        self.unithydro = uh_new
        self.params = params_new
        self.scurve = scurve_new
        return self

    def convolve(self, rainfall: pd.Series | pd.DataFrame, **kwargs: Any
                 ) -> pd.Series | pd.DataFrame:
        """
        Returns:
            pd.Series | pd.DataFrame: The resulting flood hydrograph
                after convolving the rainfall series with the unit hydrograph.

        Args:
            rainfall (array_like): Series of rain data

        Raises:
            ValueError: If the unit hydrograph is not computed inside the class

        Returns:
            (pandas.Series): flood hydrograph 
        """
        dt = rainfall.index[1]-rainfall.index[0]
        if dt != self.timestep:
            text = 'Rain series and UH time resolution must match !!'
            raise RuntimeError(text)
        if len(rainfall.shape) > 1:
            def func(col): return sg.convolve(col, self.unithydro)
            hydrograph = rainfall.apply(func, **kwargs)
        else:
            hydrograph = pd.Series(sg.convolve(rainfall, self.unithydro))
        hydrograph.index = hydrograph.index*self.timestep
        return hydrograph

    def compute(self, timestep: float, upper_tail_threshold: float = 1e-4,
                **kwargs: Any) -> Type['LumpedUnitHydrograph']:
        """
        Trigger calculation of desired unit hydrograph

        Args:
            timestep (float): Desired time step in hours.
            upper_tail_threshold (float, optional): Tolerance for the unit
                hydrograph upper tail volume accumulation. High tolerance could
                imply a unit  hydrograph that isnt strighly unitary.
                Defaults to 1e-4. 

        Raises:
            ValueError: If give the class a wrong UH kind.

        Returns:
            self: Updated Class
        """
        self.kwargs = {**self.kwargs, **kwargs}
        self.timestep = timestep
        method = self.method
        if method == 'Clark':
            uh, uh_params = self._Clark(timestep=timestep, **kwargs)
        elif method == 'SCS':
            uh, uh_params = self._SCS(timestep=timestep, **kwargs)
        elif method == 'Linsley':
            uh, uh_params = self._Linsley(**kwargs)
        elif method == 'Gray':
            uh, uh_params = self._Gray(**kwargs)
        else:
            raise ValueError(f'method="{method}" not valid!')

        uh = uh[(uh.cumsum()/uh.sum()) < 1-upper_tail_threshold]
        self.unithydro, self.params = uh, uh_params
        self.scurve = self.get_SHydrograph()
        self.update_duration(self.timestep)

        return self

    def plot(self, **kwargs: Any) -> Tuple[plt.figure, plt.axes]:
        """
        Simple accessor to plotting the unit hydrograph
        Args:
            **kwargs are given to pandas plot method
        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: 
            A tuple containing the figure and axes of the plot.
        """
        params = self.params.to_dict()
        text = [f'{key}: {val:.2f}' for key, val in params.items()]
        text = ' ; '.join(text)
        fig = self.unithydro.plot(xlabel='(hr)', ylabel='m3 s-1 mm-1',
                                  title=text, **kwargs)
        ax = plt.gca()
        return (fig, ax)
