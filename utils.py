'''

 # @ Author: Lucas Glasner (lgvivanco96@gmail.com)
 # @ Create Time: 1969-12-31 21:00:00
 # @ Modified by: Lucas Glasner, 
 # @ Modified time: 2024-05-06 16:59:31
 # @ Description:
 # @ Dependencies:
 '''

import pandas as pd

# ---------------------------------------------------------------------------- #


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
