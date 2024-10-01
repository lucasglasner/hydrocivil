## hydrocivil: RationalMethod

In this directory there is an example of how to use hydrocivil to perform fast multi-basin runoff computations using the classic "rational method". The rational method is a simple formula used in engineering projects, that holds for very small basins ($A < 20 km^2$). The method suggests that the peak runoff of a flood can be computed as: 

$$
Q = C \cdot i \cdot A

$$

where $Q$ is the peak runoff, $i$ is the precipitation intensity of a storm of duration equal to the concentration time and $A$ is the drainage area.

The notebook *sdh_rational-method.ipynb* shows an example of runoff computing for different formulations of the time of concentration and basin properties given on *params.xlsx* file. This file itself was generated using hydrocivil and a DEM (see *Geomorphometric_Analysis* example notebooks).
