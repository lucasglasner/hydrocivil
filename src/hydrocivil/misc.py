'''

 # @ Author: Lucas Glasner (lgvivanco96@gmail.com)
 # @ Create Time: 1969-12-31 21:00:00
 # @ Modified by: Lucas Glasner, 
 # @ Modified time: 2024-05-06 16:59:31
 # @ Description:
 # @ Dependencies:
 '''

import os
import pandas as pd
import numpy as np

# ------------------------------------ gis ----------------------------------- #


def chile_region(point, regions):
    """
    Given a geopandas point and region/states polygones
    this function returns the ID of the polygon where the point belongs
    (as long as an ID column exists in the region polygons)
    If an ID column doesnt exist in the region polygons this function 
    will return the polygon index.

    Args:
        point (geopandas.Series): point 
        regions (geopandas.GeoDataFrame): collection of polygons that
            represents regions or states.

    Returns:
        (object): ID of the polygon that contains the point
    """
    if 'ID' not in regions.columns:
        regions['ID'] = regions.index

    point = point.to_crs(regions.crs)
    mask = []
    for i in range(len(regions)):
        geom = regions.geometry.iloc[i]
        mask.append(point.within(geom))
    region = regions.iloc[mask, :]['ID']
    return region.item()


def raster_distribution(raster, **kwargs):
    """
    Given a raster this function computes the histogram

    Args:
        raster (xarray.Dataset): raster

    Returns:
        (pandas.Series): Histogram with data in index and pdf in values
    """
    total_pixels = raster.size
    total_pixels = total_pixels-np.isnan(raster).sum()
    total_pixels = total_pixels.item()

    values = raster.squeeze().values.flatten()
    values = values[~np.isnan(values)]
    dist, values = np.histogram(values, **kwargs)
    dist, values = dist/total_pixels, 0.5*(values[:-1]+values[1:])
    return pd.Series(dist, index=values, name=f'{raster.name}_dist')


# ----------------------------------- other ---------------------------------- #

def get_psep():
    """
    Return path separator for the current os

    Returns:
        (str): path separator
    """
    if os.name == 'nt':
        psep = '\\'
    else:
        psep = '/'
    return psep


def is_iterable(obj):
    """
    Simple function to check if an object
    is an iterable.

    Args:
        obj (any): Any python class

    Returns:
        (bool): True if iterable False if not
    """
    try:
        iter(obj)
        return True
    except:
        return False


def to_numeric(obj):
    """
    Simple function to transform object to numeric if possible

    Args:
        obj (_type_): _description_

    Returns:
        _type_: _description_
    """
    try:
        return pd.to_numeric(obj)
    except:
        return obj