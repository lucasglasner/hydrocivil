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

# ----------------------------------- PATHS ---------------------------------- #
ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(ROOT_FOLDER, 'resources')
SHYETO_PATH = os.path.join(DATA_FOLDER, 'synthetic_storms.csv')
ABZONE_PATH = os.path.join(DATA_FOLDER, 'vector', 'Zonas_ArteagaBenitez.shp')

# ----------------------------------- DATA ----------------------------------- #
SHYETO_DATA = pd.read_csv(SHYETO_PATH, index_col=0)
ABZONE_POLYGON = gpd.read_file(ABZONE_PATH)
