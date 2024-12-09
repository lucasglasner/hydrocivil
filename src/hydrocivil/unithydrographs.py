'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 1969-12-31 21:00:00
 Modified by: Lucas Glasner, 
 Modified time: 2024-05-06 16:40:29
 Description:
 Dependencies:
'''

import os
import warnings
import numpy as np
import pandas as pd
import scipy.signal as sg
import geopandas as gpd

from math import gamma
from scipy.interpolate import interp1d
from scipy.special import gamma

from .geomorphology import tc_SCS


# ----------------------------- UNIT HYDROGRAPHS ----------------------------- #


def SUH_SCS(area, mriverlen, meanslope, curvenumber, tstep, m=3.7,
            interp_kwargs={'kind': 'quadratic'}):
    """
    U.S.A Soil Conservation Service (SCS) synthetic unit hydrograph. 

    The SCS model for unit hydrograph calculation is based on a triangular unit 
    hydrograph and a dimensionless unit hydrograph. The model assumes that the
    unit hydrograph can be characterized as a function of the peak flow 'q_p'
    and peak time 't_p'. The equations are:

    t_p = 0.6*t_c + dt/2
    q_p = prf * area / t_p

    where t_c is the concentration time, dt the rain duration and unit 
    hydrograph timestep and prf the peak rate factor. The dimensionless unit
    hydrograph comes from solving the so called 'gamma equation', which is:

    q/q_p = exp(m) * ((t / t_p)**m) * exp(-m * (t / t_p))

    where 'm' is a shape parameter related to the PRF by the equation: 

    prf = C * m**(m + 1) / exp(m) / Γ(m+1)

    where Γ is the gamma function and C a constant equal to 645.33 for
    imperial units, and 0.2787 for international units. The peak rate factor 
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
        mriverlen (float): Main channel length (km)
        meanslope (float): Basin mean slope (m/m)
        curvenumber (float): Basin curve number (dimensionless)
        tstep (float): Unit hydrograph discretization time step in hours. 
        kind (str): Specifies the kind of interpolation as a string or as
            an integer specifying the order of the spline interpolator to use.
            Defaults to 'quadratic'.
        **kwargs are passed to scipy.interpolation.interp1d function.

    Returns:
        uh, (qp, tp, tb, tstep) (tuple):
            (unit hydrograph),
            (Peak runoff (L/s/km2/mm), peak time (hours), base time (hours),
            time step (hours)))

    """
    # Unit hydrograph shape
    t_shape = [np.arange(0, 2+0.1, 0.1),
               np.arange(2.2, 4+0.2, 0.2),
               np.arange(4.5, 8+0.5, 0.5)]
    t_shape = np.hstack(t_shape)

    q_shape = np.exp(m)*t_shape**m*np.exp(-m*t_shape)

    # Peak rate factor
    prf = 645.33*m**(m+1)/np.exp(m)/gamma(m+1)  # Imperial Units
    prf = prf*(0.0283/2.58/25.4)               # International Units

    # Unit hydrograph paremeters
    tp = (tc_SCS(mriverlen, meanslope, curvenumber)/60)*0.6+tstep/2
    tb = 2.67*tp
    qp = prf*area/tp
    uh = pd.Series(qp*q_shape, index=t_shape*tp)

    # Interpolate to new time resolution
    ntime = np.arange(uh.index[0], uh.index[-1]+tstep, tstep)
    f = interp1d(uh.index, uh.values,
                 fill_value='extrapolate', **interp_kwargs)
    uh = f(ntime)
    uh = pd.Series(uh, index=ntime)
    uh = uh.where(uh > 0).fillna(0)

    # Ensure that the unit hydrograph acummulates a volume of 1mm
    volume = np.trapz(uh, uh.index*3600)/1e6/area*1e3
    uh = uh/volume
    params = (qp, tp, tb, tstep, prf)
    params = pd.Series(
        params, index=['prf', 'qpeak', 'tpeak', 'tbase', 'tstep'])
    uh.name, params.name = 'SCS_m3 s-1 mm-1', 'Params_SCS'
    return uh, params


def tstep_correction(tstep, tp):
    """
    This functions checks if the selected timestep can be used
    as the unit hydrograph time resolution (unitary time) for DGA methods.

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
        warnings.warn(f'tstep exceeds tu/2, changing tstep to tu = tp/5.5')
        return tu, tp
    if (tstep < tu+0.1) and (tstep > tu-0.1):
        return tstep, tp
    else:
        tp = tp+0.25*(tstep-tu)
        return tstep, tp


def SUH_Gray(area, mriverlen, meanslope, tstep,
             interp_kwargs={'kind': 'quadratic'},
             **kwargs):
    """ 
    Gray method for Synthethic Unit Hydrograph (SUH). This method assumes a SUH
    that follows the gamma function, which assumes that the basin response is
    the same as an infinite series of lineal reservoirs. 

    Peak time and gamma_parameter (lowercase) parameters are calibrated using 
    Chilean basins. Just like the Arteaga & Benitez method, Gray SUH is only 
    valid for the III to the X Chilean political regions. 

    References:
        Manual de calculo de crecidas y caudales minimos en cuencas sin 
        informacion fluviometrica. Republica de Chile, Ministerio de Obras
        Publicas (MOP), Dirección General de Aguas (DGA) (1995). 

        Bras, R. L. (1990). Hydrology: an introduction to hydrologic science.

        ???


    Args:
        area (float): Basin area (km2)
        mriverlen (float): Main channel length (km)
        meanslope (float): Basin mean slope (m/m)
        tstep (float): Unit hydrograph target unitary time (tu) in hours. 
        interp_kwargs (dict): args to scipy.interpolation.interp1d function.
    """
    def Gray_peaktime(L, S):
        """
        Args:
            L (float): Main channel length (km)
            S (float): Basin mean slope (m/m)

        Returns:
            (float): Method suggestion for the SUH peak time
        """
        a = (np.sqrt(S) / L)**0.155
        tp = 192.5/(2.94*a-1)
        return tp/60

    def Gray_gamma_param(L, S):
        """
        Args:
            L (float): Main channel length (km)
            S (float): Basin mean slope (m/m)

        Returns:
            (float): method shape function parameter
        """
        a = (np.sqrt(S) / L)**0.155
        y = 2.68/(1-0.34*a)
        return y

    y = Gray_gamma_param(mriverlen, meanslope)
    tp = Gray_peaktime(mriverlen, meanslope)
    # tstep, tp = tstep_correction(tstep, tp)
    tstep = tp/5.5

    t_shape = np.arange(0, 10+0.05, 0.05)
    q_shape = 25*y**(y+1)*np.exp(-y*t_shape)*(t_shape)**(y)/gamma(y+1)

    uh = pd.Series(q_shape, index=t_shape*tp)
    uh = uh*area/360/(0.25*tp)
    mask = uh < 1e-4
    mask[0] = False
    uh = uh.where(~mask).dropna()
    uh.loc[uh.index[-1]+tstep] = 0

    # Interpolate to new time resolution
    ntime = np.arange(uh.index[0], uh.index[-1]+tstep, tstep)
    f = interp1d(uh.index, uh.values,
                 fill_value='extrapolate', **interp_kwargs)
    uh = f(ntime)
    uh = pd.Series(uh, index=ntime)
    uh = uh.where(uh > 0).fillna(0)

    # Ensure that the unit hydrograph acummulates a volume of 1mm
    volume = np.trapz(uh, uh.index*3600)/1e6/area*1e3
    uh = uh/volume
    params = (uh.max(), tp, tstep)
    params = pd.Series(params, index=['qpeak', 'tpeak', 'tstep'])
    uh.name, params.name = 'Gray_m3 s-1 mm-1', 'Params_Gray'
    return uh, params


# def ArteagaBenitez_zone(region):
#     """
#     Given a Chilean political region as a string, this function returns
#     the corresponding zone of the Arteaga&Benitez 1979 unit hydrograph.

#     References:
#         Manual de calculo de crecidas y caudales minimos en cuencas sin
#         informacion fluviometrica. Republica de Chile, Ministerio de Obras
#         Publicas (MOP), Dirección General de Aguas (DGA) (1995).

#         Metodo para la determinación de los hidrogramas sintéticos en Chile,
#         Arteaga F., Benitez A., División de Estudios Hidrológicos,
#         Empresa Nacional de Electricidad S.A (1985).


#     Args:
#         region (str): Chilean region as string (e.g RM, V, IV, etc)

#     Raises:
#         RuntimeError: If a wrong region is given

#     Returns:
#         (str): Method geographical zone:
#             options: "I", "II", "III" or "IV"
#     """
#     if region in ['III', 'IV', 'V', 'RM', 'VI']:
#         return 'I'
#     elif region in ['VII']:
#         return 'II'
#     elif region in ['VIII', 'IX', 'XIV', 'X', 'XVI']:
#         return 'III'
#     elif region in ['II']:
#         return 'IV'
#     elif region in ['XV', 'I']:
#         text = f'Region: {region} not strictly aviable for the method.'
#         text = text+' Using the nearest zone: "IV"'
#         warnings.warn(text)
#         return 'IV'
#     elif region in ['XI', 'XII']:
#         text = f'Region: {region} not strictly aviable for the method.'
#         text = text+' Using the nearest zone: "III"'
#         warnings.warn(text)
#         return 'III'
#     else:
#         raise RuntimeError(f'Region: "{region}" invalid')


def SUH_ArteagaBenitez(area, mriverlen, out2centroidlen, meanslope,
                       zone_AB, tstep, interp_kwargs={'kind': 'quadratic'},
                       **kwargs):
    """
    Arteaga & Benitez Synthetic Unit Hydrograph, using Linsley formulas. 

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

    *** Shape paremeters probably depend on hydrological soil properties and
    land cover use. ¿? Maybe exists a better method in 2024 ¿? 

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
        out2centroidlen (float): Distance from basin outlet
                                 to basin centroid (km)
        meanslope (float): Basin mean slope (m/m)
        zone (str): "I", "II", "III", "IV".
            Where Type IV for the deep Atacama Desert, 
                Type I for Semi-arid Chile
                Type II for Central Chile (Maule)
                Type III for Southern Chile 
        tstep (float): Unit hydrograph target unitary time (tu) in hours. 
        interp_kwargs (dict): args to scipy.interpolation.interp1d function.

    Returns:
        uh, (qpR, tpR, tbR, tuR) (tuple):
            (unit hydrograph),
            (Peak runoff (m3/s/km2/mm), peak time (hours), time step (hours)))
    """
    def SUH_ArteagaBenitez_Coefficients():
        """ 
        Returns:
            Linsley_parameters (pandas.DataFrame): 
                Parameters of the Linsley synthetic unit hydrograph
                for the different Chilean political regions based on
                Arteaga & Benitez seminal work. 
        """
        Linsley_parameters = {
            'Ct': [0.323,   0.584,   1.351,   0.386],
            'nt': [0.422,	0.327,   0.237,   0.397],
            'Cp': [144.141, 522.514, 172.775, 355.2],
            'np': [-0.796, -1.511,  -0.835,  -1.22],
            'Cb': [5.377,	1.822,	 5.428,	  2.7],
            'nb': [0.805,	1.412,	 0.717,	  1.104]
        }
        Linsley_parameters = pd.DataFrame(Linsley_parameters,
                                          index=['I', 'II', 'III', 'IV'])
        return Linsley_parameters

    coeffs = SUH_ArteagaBenitez_Coefficients().loc[zone_AB]
    tp = coeffs['Ct']
    tp = tp * (mriverlen*out2centroidlen /
               np.sqrt(meanslope))**coeffs['nt']

    # Adjust storm duration to the UH timestep
    # tstep, tpR = tstep_correction(tstep, tp)
    tpR = tp
    # tstep, tp = tstep_correction(tstep, tp)
    tstep = tp/5.5
    qpR = coeffs['Cp']*tpR**(coeffs['np'])
    tbR = coeffs['Cb']*tpR**(coeffs['nb'])

    # Unit hydrograph shape
    t_shape = np.array([0, 0.3, 0.5, 0.6, 0.75, 1, 1.3, 1.5, 1.8, 2.3, 2.7, 3])
    q_shape = np.array([0, 0.2, 0.4, 0.6, 0.80, 1, 0.8, 0.6, 0.4, 0.2, 0.1, 0])

    # Compute unit hydrograph of duration equals tu
    uh = np.array(q_shape)*qpR
    uh = pd.Series(uh, index=np.array(t_shape)*tpR)

    # Interpolate to new time resolution
    ntime = np.arange(uh.index[0], uh.index[-1] + tstep, tstep)
    f = interp1d(uh.index, uh.values,
                 fill_value='extrapolate', **interp_kwargs)
    uh = f(ntime)
    uh = pd.Series(uh, index=ntime)
    uh = uh.where(uh > 0).fillna(0)

    # Ensure that the unit hydrograph acummulates a volume of 1mm
    volume = np.trapz(uh, uh.index*3600)/1e6  # mm
    uh = uh/volume
    params = (qpR/volume*area/1e3, tpR, tbR, tstep)
    params = pd.Series(params, index=['qpeak', 'tpeak', 'tbase', 'tstep'])
    uh.name, params.name = 'A&B_m3 s-1 mm-1', 'Params_A&B'
    return uh*area/1e3, params

# -------------------------------- MAIN CLASS -------------------------------- #


class SynthUnitHydro(object):
    """
    Synthetic Unit Hydrograph class used to building unit hydrograph of 
    river basins as a function of geomorphometric and land use properties. 
    The class can be used to build the SCS unit hydrograph of any watershed
    (using SCS time of concentration) or the Chilean unit hydrographs of 
    Arteaga&Benitez (Linsley) and Grey both present in the national flood 
    manual. 

    Examples:
        #### Compute SCS UH for a storm duration of 1 hour. Show params and
        #### S-Curve
        -> suh = SynthUnitHydro(basin_params, method='SCS', timestep=1)
        -> suh.compute()
        -> suh.UnitHydroParams 
        -> suh.SHydrograph

        #### Compute convolution of unit hydrograph and a series of rainfall
        -> suh.convolve(rainfall)

        #### Change duration of unit hydrograph to 30 minutes and compute 
        #### convolution with a new series of 30 minute rainfall pulses
        -> suh.update_duration(30/60).convolve(rainfall2)
    """

    def __init__(self, basin_params, method, timestep=30/60):
        """
        Synthetic unit hydrograph (SUH) constructor.

        Args:
            method (str): Type of synthetic unit hydrograph to use. 
                Options: 'SCS', 'Arteaga&Benitez' or 'Gray'
            basin_params (dict): Dictionary with input parameters. 
            timestep (float): unit hydrograph timestep.
                Default to 30/60 hours (30min)
        """
        self.method = method
        self.basin_params = pd.Series(basin_params)
        self.timestep = timestep
        self.UnitHydro = None
        self.UnitHydroParams = None
        self.SHydrograph = None

    def __repr__(self) -> str:
        """
        What to show when invoking a SynthUnitHydro object
        Returns:
            str: Some metadata
        """
        text = f'Unit Hydrograph: {self.method}\n'
        text = text+f'Reference Storm duration: {self.timestep}\n'
        text = text+f'Parameters:\n\n{self.UnitHydroParams}'
        return text

    def get_SHydrograph(self):
        """
        This function computes the S-Curve or S-Hydrograph which is independent
        of the storm duration and can be used for computing the UH of a
        different duration.

        Returns:
            (pandas.Series): S-Unit Hydrograph. 
        """
        uh = self.UnitHydro
        sums = [uh.shift(i) for i in range(len(uh)+1)]
        S_uh = pd.concat(sums, axis=1).sum(axis=1)
        return S_uh

    def update_duration(self, duration, kind='quadratic', **kwargs):
        """
        This function uses de S-Curve to update the unit hydrograph duration
        and the respective parameters.

        Args:
            duration (float): New storm duration (equal to time resolution)
            kind (str, optional): Specifies the kind of interpolation as
            a string. Defaults to 'quadratic'.
            **kwargs are given to interpolation function

        Returns:
            self: Updated Class
        """
        time, SCurve = self.UnitHydro.index, self.SHydrograph
        new_time = np.arange(time[0], time[-1]+duration, duration)
        interp_func = interp1d(time, SCurve.values, fill_value='extrapolate',
                               kind=kind, **kwargs)
        SCurve_new = pd.Series(interp_func(new_time), index=new_time)
        uh_new = (SCurve_new-SCurve_new.shift(1).fillna(0))
        uh_new = uh_new.where(uh_new > 0).dropna()
        uh_new.loc[uh_new.index[-1]+duration] = 0
        uh_new.loc[0] = 0
        uh_new = uh_new.sort_index()

        # Ensure that the unit hydrograph acummulates a volume of 1mm
        volume = np.trapz(uh_new, uh_new.index*3600)
        volume = volume/self.basin_params['area']/1e3  # mm
        uh_new = uh_new/volume

        params_new = pd.Series([uh_new.max(), uh_new.idxmax(),
                                uh_new.index[-1], duration],
                               index=['qpeak', 'tpeak', 'tbase', 'tstep'])

        # Update
        self.timestep = duration
        self.UnitHydro = uh_new
        self.UnitHydroParams = params_new
        self.SHydrograph = SCurve_new
        return self

    def convolve(self, rainfall, **kwargs):
        """
        Solve for the convolution of a rainfall time series and the
        unit hydrograph

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
            def func(col): return sg.convolve(col, self.UnitHydro)
            hydrograph = rainfall.apply(func, **kwargs)
        else:
            hydrograph = pd.Series(sg.convolve(rainfall, self.UnitHydro))
        hydrograph.index = hydrograph.index*self.timestep
        return hydrograph

    def compute(self, method=None, interp_kwargs={'kind': 'quadratic'},
                **kwargs):
        """
        Trigger calculation of desired unit hydrograph

        Args:
            method (str): Type of synthetic unit hydrograph
                Options: SCS, Arteaga&Benitez, Gray.

        Raises:
            ValueError: If give the class a wrong UH kind.

        Returns:
            self: Updated Class
        """
        if type(method) == type(None):
            method = self.method
        if method == 'SCS':
            params = ['area', 'mriverlen', 'meanslope', 'curvenumber']
            params = self.basin_params[params]
            uh, uh_params = SUH_SCS(tstep=self.timestep,
                                    interp_kwargs=interp_kwargs,
                                    **params, **kwargs)

        elif method == 'Arteaga&Benitez':
            params = ['area', 'mriverlen', 'out2centroidlen',
                      'meanslope', 'zone_AB']
            params = self.basin_params[params]
            uh, uh_params = SUH_ArteagaBenitez(tstep=self.timestep,
                                               interp_kwargs=interp_kwargs,
                                               **params, **kwargs)

        elif method == 'Gray':
            params = ['area', 'mriverlen', 'meanslope']
            params = self.basin_params[params]
            uh, uh_params = SUH_Gray(tstep=self.timestep,
                                     interp_kwargs=interp_kwargs,
                                     **params, **kwargs)

        else:
            raise ValueError(f'method="{method}" not valid!')

        self.UnitHydro, self.UnitHydroParams = uh, uh_params
        self.SHydrograph = self.get_SHydrograph()
        self.update_duration(self.timestep)

        return self

    def plot(self, **kwargs):
        """
        Simple accessor to plotting the unit hydrograph
        Args:
            **kwargs are given to pandas plot method
        Returns:
            output of pandas plot method
        """
        params = self.UnitHydroParams.to_dict()
        text = [f'{key}: {val:.2f}' for key, val in params.items()]
        text = ' ; '.join(text)
        fig = self.UnitHydro.plot(xlabel='(hr)', ylabel='m3 s-1 mm-1',
                                  title=text, **kwargs)
        return fig
