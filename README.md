## hydrocivil: a package for hydrological methods in civil and enviromental engineering

Typical tasks related to water resources and engineering require fast calculations of hydrological phenomena such as: flood hydrographs, flood routing, evapotranspiration, infiltration, among others. With this purpose in mind, hydrocivil is presented as an alternative package to perform calculations that are usually done in tedious spreadsheets in a flexible and adjustable way. The purpose is to give tools to the engineer to calculate hydrologic processes with methods and techniques he/she deems convenient, such as different varieties of synthetic unit hydrographs, synthetic storms or basin geomorphometric parameters. The package is not intended to be a replacement for larger hydrological models (e.g. HEC-HMS), but rather a fast, customizable and automatic alternative for simple multi-basin calculations.

The package is largely oriented to Chilean national standards, however many methods originally come from the USA NCRS National Engineering Handbook. The package is 100% written in English in order to maintain consistency with the syntax and basic classes/functions of the Python language.

## Dependencies

Before installing, check your environment for the following packages:

* numpy, pandas, scipy, matplotlib
* xarray, rasterio, rioxarray
* shapely, geopandas
* networkx

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
wshed = wshed.compute_params()
wshed.plot()
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
storm.plot(plot_Losses=True, kind='bar', width=1, legend=True, Losses_kwargs={'color':'tab:purple'}, ec='k')
```

    <Axes: >

![png](image/example_storm.png)

#### Estimate the basin response (flood hydrograph)

```python
# Compute the basin SCS unit hydrograph for the storm (UH related to the storm timestep)
wshed = wshed.SynthUnitHydro(method='SCS', timestep=storm.timestep)

# Compute the flood hydrograph as the convolution of the design storm with the unit hydrograph
wshed.UnitHydro.convolve(storm.Effective_Hyetograph).plot()
```

    <Axes: >

![png](image/example_hydrograph.png)

## References

```bib
@article{NCRS_NEH630,
  title={National Engineering Handbook Part 630 - Hydrology},
  author={Natural Resources Conservation Service, United States Department of Agriculture (USDA)},
  year={}
}

@article{mcarreteras,
  title={Manual de Carreteras},
  author={Dirección de vialidad, Ministerio de Obras Públicas (MOP), Chile},
  year={2022}
}

@article{DGA_modificacioncauces,
  title={Guías metodológicas para presentación y revisión técnica de proyectos de modificación de cauces naturales y artificiales.},
  author={Dirección General de Aguas (DGA), Ministerio de Obras Públicas (MOP), Chile},
  year={2016}
}

@article{DGA_manualcrecidas,
  title={Manual de cálculo de crecidas y caudales mínimos en cuencas sin información fluviométrica},
  author={Dirección general de Aguas (DGA), Ministerio de Obras Públicas (MOP), Chile},
  year={1995},
}

```
