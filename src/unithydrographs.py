'''
 # @ HEADER: 
 # @ Author: Lucas Glasner (lgvivanco96@gmail.com)
 # @ Create Time: 1969-12-31 21:00:00
 # @ Modified by: Lucas Glasner, 
 # @ Modified time: 2024-05-06 16:40:29
 # @ Description:
 # @ Dependencies:
 '''

import warnings
import numpy as np
import pandas as pd
import scipy.signal as sg
from scipy.interpolate import interp1d


# ----------------------------- UNIT HYDROGRAPHS ----------------------------- #
def SUH_SCS(area_km2, mriverlen_km, meanslope_1, curvenumber_1,
            tstep, interp_kwargs={'kind': 'quadratic'}):
    """
    U.S.A Soil Conservation Service (SCS) synthetic unit hydrograph
    (dimensionless). 

    References:
        Bhunya, P. K., Panda, S. N., & Goel, M. K. (2011).
        Synthetic unit hydrograph methods: a critical review.
        The Open Hydrology Journal, 5(1).

        Chow Ven, T., Te, C. V., RC, M. D., & Mays, L. W. (1988).
        Applied hydrology. McGraw-Hill Book Company.

        Snyder, F. F., Synthetic unit-graphs, Trans. Am. Geophys. Union,
        vol. 19, pp. 447-454, 1938. Soil Conservation Service, Hydrology,
        sec. 4 del National Engineering Handhook. Soil Conservation Service,
        U. S. Department of Agriculture, Washington, D.C., 1972

        Snyder, F. F. (1938). Synthetic unit‐graphs. Eos,
        Transactions American Geophysical Union, 19(1), 447-454.

    Args:
        area_km2 (float): Basin area (km2)
        mriverlen_km (float): Main channel length (km)
        meanslope_1 (float): Basin mean slope (m/m)
        curvenumber_1 (float): Basin curve number (dimensionless)
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
    def SCS_lagtime(L, CN, S):
        """
        This functions returns the lagtime of the SCS unit hydrograph.

        Args:
            L (float): Main channel length (km)
            CN (float): Curve number (dimensionless)
            S (float): Mean slope (m/m)

        Returns:
            lagtime (float): basin lagtime (hours)
        """
        a = (L*1e3)**0.8*(2540-22.86*CN)**0.7
        b = (14104*CN**0.7*S**0.5)
        lagtime = a/b
        return lagtime
    # Unit hydrograph shape
    t_shape = [np.arange(0, 2+0.1, 0.1),
               np.arange(2.2, 4+0.2, 0.2),
               np.arange(4.5, 5+0.5, 0.5)]
    t_shape = np.hstack(t_shape)

    q_shape = np.array(
        [0, 0.03, 0.1, 0.19, 0.31, 0.47, 0.66, 0.82, 0.93, 0.99, 1,
            0.99, 0.93, 0.86, 0.78, 0.68, 0.56, 0.46, 0.39, 0.33, 0.28, 0.207,
         0.147, 0.107, 0.077, 0.055, 0.040, 0.029, 0.021, 0.015, 0.011,
         0.005, 0])
    # Unit hydrograph paremeters
    tL = SCS_lagtime(mriverlen_km, curvenumber_1, meanslope_1)
    tp = tL+tstep/2
    tb = 2.67*tp
    qp = 0.208*area_km2/tp
    uh = pd.Series(qp*q_shape, index=t_shape*tp)

    # Interpolate to new time resolution
    ntime = np.arange(uh.index[0], uh.index[-1]+tstep, tstep)
    f = interp1d(uh.index, uh.values,
                 fill_value='extrapolate', **interp_kwargs)
    uh = f(ntime)
    uh = pd.Series(uh, index=ntime)
    uh = uh.where(uh > 0).fillna(0)

    # Ensure that the unit hydrograph acummulates a volume of 1mm
    volume = np.trapz(uh, uh.index*3600)/1e6/area_km2*1e3
    uh = uh/volume
    params = (qp, tp, tb, tstep)
    params = pd.Series(params, index=['qpeak', 'tpeak', 'tbase', 'tstep'])
    return uh, params


def SUH_ArteagaBenitez(area_km2, mriverlen_km, out2centroidlen_km, meanslope_1,
                       zone, tstep, interp_kwargs={'kind': 'quadratic'}):
    """
    Arteaga & Benitez Synthetic Unit Hydrograph, using Linsley formulas. 

    Linsley unit hydrograph (UH) is a similar formulation of the well known
    Snyder UH. The difference is that Linsley's UH uses a different formula
    for the peak flow like this: 

        tp = Ct * (L * Lg / sqrt(S)) ^ nt
        qp = Cp * tp ^ np
        tb = Cb * tp ^ nb

    Where L, Lg and S are the following geomorphological properties:
    basin main channel length, distance between the basin outlet and centroid,
    and basin mean slope. 

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
        area_km2 (float): Basin area (km2)
        mriverlen_km (float): Main channel length (km)
        out2centroidlen_km (float): Distance from basin outlet
                                 to basin centroid (km)
        meanslope_1 (float): Basin mean slope (m/m)
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
            'Ct': [0.323, 0.584,	1.351, 0.386],
            'nt': [0.422,	0.327, 0.237, 0.397],
            'Cp': [144.141, 522.514, 172.775, 355.2],
            'np': [-0.796, -1.511, -0.835, -1.22],
            'Cb': [5.377,	1.822,	5.428,	2.7],
            'nb': [0.805,	1.412,	0.717,	1.104]
        }
        Linsley_parameters = pd.DataFrame(Linsley_parameters,
                                          index=['I', 'II', 'III', 'IV'])
        return Linsley_parameters

    def tstep_correction(tstep, tu):
        """
        This functions checks if the selected timestep can be used
        as the unit hydrograph time resolution (unitary time). 

        Args:
            tstep (float): Desired time step in hours
            tu (float): Raw unit hydrograph unitary time (tu) in hours. 

        Raises:
            RuntimeError: If selected tstep exceeds a 10% change of 
                unit hydrograph storm duration/unitary time. 

        Returns:
            float: Fixed timestep in hours
        """
        tR = np.round(tu/tstep, 1)*tstep
        if ~((tR-tu) < 0.1 and (tR-tu) > -0.1):
            warnings.warn(f'tu: {tR:.3f} - (tR-tu) exceeds 10%, changing tstep\
                           to tu.')
        else:
            tR = tu
        return tR

    coeffs = SUH_ArteagaBenitez_Coefficients().loc[zone]
    tp = coeffs['Ct']
    tp = tp * (mriverlen_km*out2centroidlen_km /
               np.sqrt(meanslope_1))**coeffs['nt']
    tu = tp/5.5
    if tu > tstep*1.1:
        # raise RuntimeError(f'tu = {tu:.3f} cant exceed 1.1 * (timestep)!!')
        tstep = tu

    # Adjust storm duration to the UH timestep
    tR = tstep_correction(tstep, tu)
    tpR = tp+(tR-tu)/4
    qpR = coeffs['Cp']*tpR**(coeffs['np'])
    tbR = coeffs['Cb']*tpR**(coeffs['nb'])
    tuR = tR

    # Unit hydrograph shape
    t_shape = np.array([0, 0.3, 0.5, 0.6, 0.75, 1, 1.3, 1.5, 1.8, 2.3, 2.7, 3])
    q_shape = np.array([0, 0.2, 0.4, 0.6, 0.80, 1, 0.8, 0.6, 0.4, 0.2, 0.1, 0])

    # Compute unit hydrograph of duration equals tu
    uh = np.array(q_shape)*qpR
    uh = pd.Series(uh, index=np.array(t_shape)*tpR)

    # Interpolate to new time resolution
    ntime = np.arange(uh.index[0], uh.index[-1]+tuR, tuR)
    f = interp1d(uh.index, uh.values,
                 fill_value='extrapolate', **interp_kwargs)
    uh = f(ntime)
    uh = pd.Series(uh, index=ntime)
    uh = uh.where(uh > 0).fillna(0)

    # Ensure that the unit hydrograph acummulates a volume of 1mm
    volume = np.trapz(uh, uh.index*3600)/1e6  # mm
    uh = uh/volume
    params = (qpR/volume*area_km2/1e3, tpR, tbR, tuR)
    params = pd.Series(params, index=['qpeak', 'tpeak', 'tbase', 'tstep'])

    return uh*area_km2/1e3, params

# -------------------------------- MAIN CLASS -------------------------------- #


class SynthUnitHydro(object):
    def __init__(self, basin_params, method, timestep):
        """
        Synthetic unit hydrograph (SUH) constructor.

        Args:
            method (str): Type of synthetic unit hydrograph to use. 
                Options: 'SCS', 'Arteaga&Benitez', 
            basin_params (dict): Dictionary with input parameters. 
            timestep (float): unit hydrograph timestep. 
        """
        self.method = method
        self.basin_params = pd.Series(basin_params)
        self.timestep = timestep
        self.UnitHydro = None
        self.UnitHydroParams = None

    def compute(self, interp_kwargs={'kind': 'quadratic'}):
        """
        Trigger calculation of desired unit hydrograph

        Raises:
            ValueError: If give the class a wrong UH kind.

        Returns:
            (tuple): Unit hydrograph values and parameters.
        """
        if self.method == 'SCS':
            params = ['area_km2', 'mriverlen_km',
                      'meanslope_1', 'curvenumber_1']
            params = self.basin_params[params]
            uh, uh_params = SUH_SCS(tstep=self.timestep,
                                    interp_kwargs=interp_kwargs,
                                    **params)
            uh.name, uh_params.name = 'SCS_m3 s-1 mm-1', 'Params_SCS'

        elif self.method == 'Arteaga&Benitez':
            params = ['area_km2', 'mriverlen_km', 'out2centroidlen_km',
                      'meanslope_1', 'zone']
            params = self.basin_params[params]
            uh, uh_params = SUH_ArteagaBenitez(tstep=self.timestep,
                                               interp_kwargs=interp_kwargs,
                                               **params)
            uh.name, uh_params.name = 'A&B_m3s-1 mm-1', 'Params_A&B'

        else:
            raise ValueError(f'method="{self.method}" not valid!')

        self.UnitHydro, self.UnitHydroParams = uh, uh_params
        self.timestep = uh_params.tstep
        return self.UnitHydro, self.UnitHydroParams

    def UH_cumulative(self, duration):
        """
        Given a storm duration this function computes the S-Unit Hydrograph
        which is independent of storm duration and can be used for computing
        the UH of a different duration.

        Args:
            duration (float): Unit hydrograph duration.

        Returns:
            (pandas.Series): S-Unit Hydrograph. 
        """
        tstep = np.diff(self.UnitHydro.index)[0]
        shifts = int(duration/tstep)
        n_repeats = int((len(self.UnitHydro)/shifts+1))
        sums = [self.UnitHydro.shift(shifts*i) for i in range(n_repeats)]
        cumsum = pd.concat(sums, axis=1).sum(axis=1)/duration
        ntime = np.arange(0, cumsum.index[-1]*2+tstep, tstep)
        cumsum = cumsum.reindex(ntime).ffill()
        return cumsum

    def convolve(self, rainfall):
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
        if type(self.UnitHydro) == type(None):
            raise ValueError('Compute the unit hydrograph first!')
        else:
            hydrograph = pd.Series(sg.convolve(rainfall, self.UnitHydro))
            hydrograph.index = hydrograph.index*self.timestep
            return hydrograph
