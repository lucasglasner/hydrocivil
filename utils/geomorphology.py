'''
 # @ Author: Lucas Glasner (lgvivanco96@gmail.com)
 # @ Create Time: 1969-12-31 21:00:00
 # @ Modified by: Lucas Glasner, 
 # @ Modified time: 2024-05-06 09:56:20
 # @ Description:
 # @ Dependencies:
 '''

import pandas as pd
import numpy as np

# -------------------- Concentration time for rural basins ------------------- #


def tc_SCS(basin_mriverlen_km, basin_meanslope_, curve_number_):
    """
    USA Soil Conservation Service (SCS) method.
    Valid for rural basins ¿?.

    Reference:
        ???

    Args:
        basin_mriverlen_km (float): Main river length in (km)
        basin_meanslope_ (float): Basin mean slope in m/m
        curve_number_ (float): Basin curve number (dimensionless)

    Returns:
        Tc (float): Concentration time (minutes)
    """
    a = ((1000/curve_number_)-9)**0.7
    b = basin_mriverlen_km**0.8/(basin_meanslope_*100)**0.5
    Tc = 3.42*a*b
    return Tc


def tc_kirpich(basin_mriverlen_km, basin_deltaheights_m):
    """
    Kirpich equation method.
    Valid for small and rural basins ¿?.

    Reference:
        ???

    Args:
        basin_mriverlen_km (float): Main river length in (km)
        basin_deltaheights_m (float): Difference between minimum and maximum
            basin elevation (m)

    Returns:
        Tc (float): Concentration time (minutes)
    """
    Tc = ((1000*basin_mriverlen_km)**1.15)/(basin_deltaheights_m**0.385)/51
    return Tc


def tc_giandotti(basin_mriverlen_km, basin_meanheight, basin_area_km2):
    """
    Giandotti equation method.
    Valid for small basins (< 20km2) with high slope (>10%) ¿?. 

    Reference:
        ???

    Args:
        basin_mriverlen_km (float): Main river length in (km)
        basin_meanheight (float): Basin mean height (meters)
        basin_area_km2 (float): Basin area (km2)

    Returns:
        Tc (float): Concentration time (minutes)
    """
    a = (4*basin_area_km2**0.5+1.5*basin_mriverlen_km)
    b = (0.8*basin_meanheight**0.5)
    Tc = 60*a/b
    return Tc


def tc_california(basin_mriverlen_km, basin_deltaheights_m):
    """
    California Culverts Practice (1942) equation.
    Valid for mountain basins ¿?.

    Reference: 
        ???

    Args:
        basin_mriverlen_km (float): Main river length in (km)
        basin_deltaheights_m (float): Difference between minimum and maximum
            basin elevation (m)

    Returns:
        Tc (float): Concentration time (minutes)

    """
    Tc = 57*(basin_mriverlen_km**3/basin_deltaheights_m)**0.385
    return Tc


def tc_spain(basin_mriverlen_km, basin_meanslope_):
    """
    Equation of Spanish/Spain regulation.

    Reference:
        ???

    Args:
        basin_mriverlen_km (float): Main river length in (km)
        basin_meanslope_ (float): Basin mean slope in m/m

    Returns:
        Tc (float): Concentration time (minutes)
    """
    Tc = 18*(basin_mriverlen_km**0.76)/((basin_meanslope_*100)**0.19)
    return Tc
