'''
 Author: Lucas Glasner (lgvivanco96@gmail.com)
 Create Time: 2024-11-03 21:34:05
 Modified by: Lucas Glasner, 
 Modified time: 2024-11-06 09:56:20
 Description:
 Dependencies:
'''

import os
import pandas as pd
import geopandas as gpd

try:
    from . import wb_tools
except:
    _has_whitebox = False
else:
    _has_whitebox = True

# ----------------------------------- PATHS ---------------------------------- #
ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'resources')
SHYETO_PATH = os.path.join(DATA_FOLDER, 'synthetic_storms.csv')
AB_ZONEPATH = os.path.join(DATA_FOLDER, 'vector', 'ZONAS_LINSLEYCHILE.shp')
CHILE_GRAYZONEPATH = os.path.join(DATA_FOLDER, 'vector', 'ZONAS_GRAYCHILE.shp')

# ---------------------------- DATA AND PARAMETERS --------------------------- #
SHYETO_DATA = pd.read_csv(SHYETO_PATH, index_col=0)

CHILE_UH_LINSLEYPOLYGONS = gpd.read_file(AB_ZONEPATH)
CHILE_UH_LINSLEYPARAMS = {
    'C_t': [0.323,   0.584,   1.351,   0.386],
    'n_t': [0.422,	0.327,   0.237,   0.397],
    'C_p': [144.141, 522.514, 172.775, 355.2],
    'n_p': [-0.796, -1.511,  -0.835,  -1.22],
    'C_b': [5.377,	1.822,	 5.428,	  2.7],
    'n_b': [0.805,	1.412,	 0.717,	  1.104]
}
CHILE_UH_LINSLEYPARAMS = pd.DataFrame(CHILE_UH_LINSLEYPARAMS).T
CHILE_UH_LINSLEYPARAMS.columns = ['I', 'II', 'III', 'IV']

CHILE_UH_GRAYPOLYGONS = gpd.read_file(CHILE_GRAYZONEPATH)
CHILE_UH_GRAYPARAMS = {'a': 24.48, 'b': 0.155}
GDAL_EXCEPTIONS = True
