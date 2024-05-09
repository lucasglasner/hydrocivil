'''
 # @ Author: Lucas Glasner (lgvivanco96@gmail.com)
 # @ Create Time: 1969-12-31 21:00:00
 # @ Modified by: Lucas Glasner, 
 # @ Modified time: 2024-05-06 16:40:13
 # @ Description:
 # @ Dependencies:
 '''

from utils import is_iterable

# -------- curve number correction for antecedent moisture conditions -------- #


def CN_correction(CN_II, AMC):
    """
    This function changes the curve number value according to antecedent
    moisture conditions (AMC)...

    Reference: 
        Ven Te Chow (1988), Applied Hydrology. MCGrow-Hill
        Soil Conservation Service, Urban hydrology for small watersheds,
        tech. re/. No. 55, U. S. Dept. of Agriculture, Washington, D.E:., 1975.

    Args:
        CN_II (float): curve number under normal condition
        AMC (str): Antecedent moisture condition.
            Options: 'dry', 'wet' or 'normal'

    Raises:
        RuntimeError: If AMC is different than 'dry', 'wet' or 'normal'

    Returns:
        CN_I or CN_III (float): _description_
    """
    if AMC == 'dry':
        CN_I = 4.2*CN_II/(10-0.058*CN_II)
        return CN_I
    elif AMC == 'normal':
        return CN_II
    elif AMC == 'wet':
        CN_III = 23*CN_II/(10+0.13*CN_II)
        return CN_III
    else:
        text = f'AMC="{AMC}"'
        text = text+' Moisture condition can only be "dry, "normal" or "wet"'
        raise RuntimeError(text)


def cum_abstraction(pr, CN):
    """
    Compute accumulated abstractions/infiltrations

    Args:
        pr (float): precipitation
        CN (float): curve number

    Returns:
        (float): mm infiltrated into soil
    """
    # S = maximum infiltration given soil type and cover (CN)
    # I0 = 0.2 * S infiltration until pond formation
    S = (25400-254*CN)/CN
    if pr > 0.2*S:
        a = 0.2*S
    else:
        a = pr
    if pr >= 0.2*S:
        b = (S*(pr-0.2*S))/(pr+0.8*S)
    else:
        b = 0
    return a+b
