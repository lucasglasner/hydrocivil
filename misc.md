```python
from hydrocivil.misc import load_example_data
from hydrocivil.watersheds import RiverBasin
from hydrocivil.rain import DesignStorm
```

###### Compute basin properties


```python
# ---------------------- Load example data (or your own) --------------------- #

# dem = rxr.open_rasterio('/path/to/dem.tif')
# curvenumber = rxr.open_rasterio('path/to/cn.tif')
# rivernetwork = gpd.read_file('path/to/rivers.shp')
# basin_polygon = gpd.read_file('path/to/basin.shp')
basin, rnetwork, dem, cn = load_example_data()

# Create RiverBasin object and compute properties
wshed = RiverBasin('Example', basin, rnetwork, dem, cn)
wshed = wshed.compute_params()
wshed.plot()
```




    <Axes: title={'left': 'Example'}>




    
![png](misc_files/misc_2_1.png)
    


###### Create an hypothetical storm


```python
# Create a 100 milimeter, 24 hours duration, SCS type I storm with pulses every 30 minutes
storm = DesignStorm('SCS_I24')
storm = storm.compute(timestep=0.5, duration=24, rainfall=100)
# Use SCS method for abstractions with the watershed average curve number
storm = storm.infiltrate(method='SCS', cn=wshed.params.loc['curvenumber'].item())
storm.Hyetograph.plot()
storm.Effective_Hyetograph.plot()
```




    <Axes: >




    
![png](misc_files/misc_4_1.png)
    


###### Estimate the basin response (flood hydrograph)


```python
# Compute the basin SCS unit hydrograph for the storm (UH related to the storm timestep)
wshed = wshed.SynthUnitHydro(method='SCS', timestep=storm.timestep)

# Compute the flood hydrograph as the convolution of the design storm with the unit hydrograph
wshed.UnitHydro.convolve(storm.Effective_Hyetograph).plot()
```




    <Axes: >




    
![png](misc_files/misc_6_1.png)
    

