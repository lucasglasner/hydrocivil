## hydrocivil: a package for hydrological methods in civil and enviromental engineering

Typical tasks related to water resources and engineering require quick calculations of hydrological phenomena such as: storm hyetographs, soil infiltration, flood hydrographs, flood routing through channels or reservoirs, etc. With this purpose in mind, the package is presented as an alternative to perform calculations that are usually done in tedious spreadsheets in a fast pythonic way. The purpose is to give tools to the engineer to calculate hydrologic processes with methods and techniques he/she deems convenient, such as different varieties of synthetic unit hydrographs, synthetic storms or basin geomorphometric parameters. The package isnt intended to be a replacement of larger hydrological models (e.g. HEC-HMS), but rather a fast customizable and automatic alternative for multi-basin calculations.

The package is largely oriented to Chilean national standards, however many methods originally come from the USA NCRS National Engineering Handbook.

## Dependencies

Before installing, check your environment for the following packages:

* numpy, pandas, scipy, matplotlib
* xarray, rasterio, rioxarray
* shapely, geopandas
* networkx
* whitebox_workflows (optional)

## Installation

Currently the package can only be installed via pip:

```shell
pip install --force-reinstall --no-deps hydrocivil
```

## Example Use

```python
from hydrocivil.misc import load_example_data
from hydrocivil.watersheds import RiverBasin
from hydrocivil.rain import RainStorm
```

#### Compute basin properties

```python
# ---------------------- Load example data (or your own) --------------------- #

# dem = rxr.open_rasterio('/path/to/dem.tif')
# curvenumber = rxr.open_rasterio('path/to/cn.tif')
# rivernetwork = gpd.read_file('path/to/rivers.shp')
# basin_polygon = gpd.read_file('path/to/basin.shp')
basin, rnetwork, dem, cn = load_example_data()

# Create RiverBasin object and compute properties
wshed = RiverBasin('Example', basin, rnetwork, dem, cn, amc='wet')
wshed = wshed.compute_params()  # <- compute geomorphological parameters (SI units)
wshed.plot() # Check results (e.g basin polygon, identified main river, etc)
```

    <Axes: title={'left': 'Example'}>

 ![png](image/wshed_plot_outputexample.png)

#### Create an hypothetical storm

```python
# Create a 100 milimeter, 24 hours duration, SCS type I storm with pulses every 30 minutes
storm = RainStorm('SCS_I24')
storm = storm.compute(timestep=0.5, duration=24, rainfall=100)
# Use SCS method for abstractions with the watershed average curve number
storm = storm.infiltrate(method='SCS', cn=wshed.params.loc['curvenumber'].item())

storm.pr.to_series().plot(kind='bar', width=1, ec='k')
storm.losses.to_series().plot(kind='bar', width=1, color='tab:purple', ec='k')
```

    <Axes: >

![png](image/example_storm.png)

#### Estimate the basin response (flood hydrograph)

```python
# Compute the basin SCS unit hydrograph for the storm (UH related to the storm timestep)
wshed = wshed.SynthUnitHydro(method='SCS', timestep=storm.timestep) # By default this uses the 484 SCS unit hydrograph. 

# Compute the flood hydrograph as the convolution of the design storm with the unit hydrograph
wshed.UnitHydro.convolve(storm.Effective_Hyetograph).plot()
```

    <Axes: >

![png](image/example_hydrograph.png)
