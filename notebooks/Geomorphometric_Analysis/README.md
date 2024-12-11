## hydrocivil: Geomorphometric analysis

In this directory there is a simple workflow for a geomorphometric analysis of river basins. The work flow consists on the following:

+ Download a raw digital elevation model (DEM) from opentopo web service. In *querydem.ipynb* there is an example of how to. Note an opentopo api key is needed, check [https://opentopography.org/blog/introducing-api-keys-access-opentopography-global-datasets]() for more details.
+ Once the DEM is downloaded as a *.tif* file in *BasinDelineation.ipynb* there are examples of how to perform basin delineation using whitebox GIS tools and a point layer of basin outlets.
+ With the basin polygons and the DEM the notebook *Geomorphology.ipynb* shows an example of how to use hydrocivil to compute geomorphological properties of the watersheds.
