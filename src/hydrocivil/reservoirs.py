"""
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 2025-01-20 15:17:43
 Modified by: Lucas Glasner,
 Modified time: 2025-01-20 15:17:43
 Description: Main class for reservoir and lake routing
 Dependencies:
"""

import pandas as pd
import numpy as np

from typing import Union, Type, Tuple
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from .global_vars import GRAVITY


# class Spillway(object):
#     def __init__(self, fid: Union[str, int, float], kind: str):
#         self.fid = fid
#         self.g = GRAVITY
#         self.kind = kind

class Reservoir:
    """
    A class representing a reservoir model that simulates water level and
    discharge over time.

    Attributes:
        fid (int): Unique identifier for the reservoir.
        stage_area (pd.Series): A pandas Series representing the relationship
            between stage (index) and surface area (values).
        stage_discharge (pd.Series): A pandas Series representing the
            relationship between stage (index) and discharge (values).
    """

    def __init__(self, fid: Union[int, float, str],
                 stage_area: pd.Series = None,
                 stage_storage: pd.Series = None,
                 stage_discharge: pd.Series = None) -> Type["Reservoir"]:
        """
        Initializes the Reservoir object.

        Args:
            fid (int): Reservoir identifier.
            stage_area (pd.Series): Stage-area relationship, where the index is
                stage (m) and values are area (m2). Defaults to None.
            stage_storage (pd.Series): Stage-storage relationship, where the
                index is stage (m) and values are volume (m3). Defaults to None.
            stage_discharge (pd.Series): Stage-discharge relationship, where
                index is stage (m) and values are discharge (m3/s). Defaults to
                None.
        """
        self.fid = fid
        self.stage_area = stage_area
        self.stage_storage = stage_storage
        self.stage_discharge = stage_discharge
        self.Q_series = None
        self.h_series = None

    def _series2func(self, series: pd.Series, **kwargs) -> interp1d:
        """
        Converts a pandas Series into an interpolating function.

        Args:
            series (pd.Series): Series where index represents the independent
                variable and values the dependent variable.
            **kwargs: Additional arguments for scipy.interpolate.interp1d.

        Returns:
            interp1d: A function that interpolates or extrapolates the
                given series.
        """
        return interp1d(series.index, series.values, fill_value='extrapolate',
                        **kwargs)

    def _solve4_stagearea(self, time: np.ndarray, inflow: pd.Series,
                          h0: float = 0.0, **kwargs
                          ) -> Tuple[pd.Series, pd.Series]:
        """
        Computes the reservoir stage and discharge over time using the water
        balance equation:

            d(A * h)/dt = Qi - Qo
            dA/dt * h + A * dh / dt = Qi - Qo
            dA/dh * dh/dt * h + A * dh/dt = Qi - Qo
            dhdt = (Qi - Qo) / (A + h * dA/dh)

        where:
            A(h) is the stage-area function (hypsometry of the reservoir)
            h(t) is the water stage or level in meters
            Qi(t) is the input hydrograph in m3/s
            Qo(h) is the stage-discharge curve which depends on the reservoir
                outlet characteristics (e.g spillway, natural channel, etc).

        Args:
            time (np.ndarray): Array of time steps for the simulation.
            inflow (pd.Series): A pandas Series where index is time and values
                are inflow rates (m3/s).
            h0 (float, optional): Initial water level (m). Defaults to 0.0.
            **kwargs: Additional arguments for interpolation.

        Returns:
            Tuple[pd.Series, pd.Series]:
                - A pandas Series of water levels (stage) over time.
                - A pandas Series of corresponding discharge values.
        """
        dAdh = np.gradient(self.stage_area.values, self.stage_area.index)
        dAdh = pd.Series(dAdh, index=self.stage_area.index)

        _stage_area = self._series2func(self.stage_area, **kwargs)
        _stage_area_derivative = self._series2func(dAdh, **kwargs)
        _inflow = self._series2func(inflow, **kwargs)
        _outflow = self._series2func(self.stage_discharge, **kwargs)

        def _dhdt(h: float, t: float) -> float:
            a = (_inflow(t) - _outflow(h))
            b = (_stage_area(h)+h*_stage_area_derivative(h))
            return a / b

        h = odeint(_dhdt, h0, time).squeeze()
        h_series = pd.Series(h, index=time)
        Qs_series = h_series.map(lambda hh: _outflow(hh))
        return h_series, Qs_series

    def _solve4_stagestorage(self, time: np.ndarray, inflow: pd.Series,
                             h0: float = 0.0, **kwargs
                             ) -> Tuple[pd.Series, pd.Series]:
        """
        Computes the reservoir stage and discharge over time using the water
        balance equation:

            dS/dt = (Qi - Qo)
            dS/dh * dh/dt = (Qi - Qo)
            dh/dt = (Qi - Qo) / (dS/dh)

        where:
            S(h) is the stage-storage function (capacity of the reservoir)
            h(t) is the water stage or level in meters
            Qi(t) is the input hydrograph in m3/s
            Qo(h) is the stage-discharge curve which depends on the reservoir
                outlet characteristics (e.g spillway, natural channel, etc).

        Args:
            time (np.ndarray): Array of time steps for the simulation.
            inflow (pd.Series): A pandas Series where index is time and values
                are inflow rates (m3/s).
            h0 (float, optional): Initial water level (m). Defaults to 0.0.
            **kwargs: Additional arguments for interpolation.

        Returns:
            Tuple[pd.Series, pd.Series]:
                - A pandas Series of water levels (stage) over time.
                - A pandas Series of corresponding discharge values.
        """
        dVdh = np.gradient(self.stage_storage.values, self.stage_storage.index)
        dVdh = pd.Series(dVdh, index=self.stage_storage.index)

        _stage_storage_derivative = self._series2func(dVdh, **kwargs)
        _inflow = self._series2func(inflow, **kwargs)
        _outflow = self._series2func(self.stage_discharge, **kwargs)

        def _dhdt(h: float, t: float) -> float:
            a = (_inflow(t) - _outflow(h))
            b = (_stage_storage_derivative(h))
            return a / b

        h = odeint(_dhdt, h0, time).squeeze()
        h_series = pd.Series(h, index=time)
        Qs_series = h_series.map(lambda hh: _outflow(hh))
        return h_series, Qs_series

    def compute(self, time: np.ndarray, inflow: pd.Series,
                h0: float, **kwargs) -> Tuple[pd.Series, pd.Series]:
        """
        Computes the reservoir stage and outflow discharge over time using
        the water balance equation.

        Args:
            time (np.ndarray): Array of time steps for the simulation.
            inflow (pd.Series): A pandas Series where index is time and values
                are inflow rates (m3/s).
            h0 (float, optional): Initial water level (m).
            **kwargs: Additional arguments for interpolation.

        Returns:
            Tuple[pd.Series, pd.Series]:
                - A pandas Series of water levels (stage) over time.
                - A pandas Series of corresponding discharge values.
        """
        if (self.stage_storage is not None):
            h, Qs = self._solve4_stagestorage(time, inflow, h0, **kwargs)
        elif (self.stage_area is not None):
            h, Qs = self._solve4_stagearea(time, inflow, h0, **kwargs)
        else:
            text = 'To route hydrograph through reservoir a stage-storage '
            text += 'or stage-area curve must be aviable.'
            raise RuntimeError(text)
        self.h = h
        self.Q_series = Qs
        return h, Qs
