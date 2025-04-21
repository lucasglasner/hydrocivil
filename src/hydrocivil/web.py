'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 1969-12-31 21:00:00
 Modified by: Lucas Glasner, 
 Modified time: 2024-03-21 13:33:18
 Description:
 Dependencies:
'''

import requests


def OpenTopo(DATASET: str, ROI: tuple, OUTPUTPATH: str,
             API_KEY: str = 'YOURKEY') -> requests.models.Response:
    """
    This function makes a request to the OpenTopo server and downloads
    DEM data for a small region. 

    Args:
        DATASET (str): Any of this:
            "SRTMGL3" (SRTM GL3 90m)
            "SRTMGL1" (SRTM GL1 30m)
            "SRTMGL1_E" (SRTM GL1 Ellipsoidal 30m)
            "AW3D30" (ALOS World 3D 30m)
            "AW3D30_E" (ALOS World 3D Ellipsoidal, 30m)
            "SRTM15Plus" (Global Bathymetry SRTM15+ V2.1 500m)
            "NASADEM" (NASADEM Global DEM)
            "COP30" (Copernicus Global DSM 30m)
            "COP90" (Copernicus Global DSM 90m)
            "EU_DTM" (DTM 30m)
            "GEDI_L3" (DTM 1000m)
            "GEBCOIceTopo" (Global Bathymetry 500m)
            "GEBCOSubIceTopo" (Global Bathymetry 500m)

        ROI (tuple): Tuple with coordinates (lonmin, lonmax, latmin, latmax)

        OUTPUTPATH (str): Path to the output file e.g "./blabla/dem.tif"

        API_KEY (str, optional): Your OpenTopo API Key.
            Check https://opentopography.org/developers for details
            Defaults to 'YOURKEY'.

    Returns:
        str: server answer to get request. 
    """
    if API_KEY == 'YOURKEY':
        apiurl = 'https://opentopography.org/developers'
        raise RuntimeError(
            f'Please provide the server API key (Check {apiurl} for details)')
    lonmin, lonmax, latmin, latmax = ROI
    params = {
        'demtype': f'{DATASET}',
        'south': f'{latmin}',
        'north': f'{latmax}',
        'west': f'{lonmin}',
        'east': f'{lonmax}',
        'outputFormat': 'GTiff',
        'API_Key': f'{API_KEY}',
    }
    url = 'https://portal.opentopography.org/API/globaldem'
    response = requests.get(url, params=params)
    with open(f'{OUTPUTPATH}', 'wb') as f:
        f.write(response.content)
    return response
