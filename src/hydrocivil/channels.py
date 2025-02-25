"""
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 2025-01-20 15:17:43
 Modified by: Lucas Glasner,
 Modified time: 2025-01-20 15:17:43
 Description: Main class for channel routing
 Dependencies:
"""

import pandas as pd
import numpy as np
import geopandas as gpd

from typing import Union, Type, Tuple
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from .global_vars import GRAVITY


class RiverReach:
    """
    A class representing a channel model that simulates basic river routing
    and 1D hydraulics.

    Attributes:
        fid (int): Unique identifier for the channel.

    """

    def __init__(self, fid: Union[int, float, str],
                 method: str):
        """
        Initializes the RiverReach object.

        Args:
            fid (int): Channel identifier.
            method (str): Routing method:
                Options: 'Lag', 'NormalDepth', 'Muskingum', 'MuskingumCunge'
        """
        self.fid = fid
        self.method = method
        self.inflow = None
        self.outflow = None

    def _Lag(self, inflow: pd.Series, lag: float, **kwargs) -> pd.Series:
        """
        lag a discharge time series by a continuous (float) amount using
        interpolation.

        Parameters:
        inflow (pd.Series): The time series with numeric index representing
            time.
        lag (float): The amount of lag (can be positive or negative). Must be
            in the time series index units to make sense.

        Returns:
        pd.Series: The shifted time series.
        """
        time = inflow.index.values  # Convert index to NumPy array
        new_time = time + lag       # Shift time indices
        lagged_values = np.interp(time, new_time, inflow.values,  # Interpolate
                                  **kwargs)
        return pd.Series(lagged_values, index=time)

    def _NormalDepth():
        """
        """
        pass

    def _Muskingum():
        """
        """
        pass

    def _MuskingumCunge():
        """
        """
        pass

    def compute(self, inflow: pd.Series, **kwargs):
        """
        """
        self.inflow = inflow
        if self.method == 'Lag':
            self.outflow = self._Lag(inflow=inflow, **kwargs)
            return self.outflow
        elif self.method == 'NormalDepth':
            pass
        elif self.method == 'Muskingum':
            pass
        elif self.method == 'MuskingumCunge':
            pass
        else:
            text = f'Unknown routing method: "{self.method}"'
            raise RuntimeError(text)
