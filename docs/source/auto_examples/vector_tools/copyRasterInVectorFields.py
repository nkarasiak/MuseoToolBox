# -*- coding: utf-8 -*-
"""
Copy raster values in vector fields then read vector
======================================================

This example shows how to extract from polygons or points
each pixel centroid located in the vector (polygons/points)
and how to extract and save band values in vector fields.
 
This tool is made to avoid using raster everytime you need
to learn and predict a model."""

##############################################################################
# Import librairies
# -------------------------------------------

import museotoolbox as mtb
from matplotlib import pyplot as plt

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = mtb.datasets.getHistoricalMap()
outVector='/tmp/vector_withROI.gpkg'


###############################################################################
# .. note::
#    There is no need to specify a bandPrefix. 
#    If bandPrefix is not specified, scipt will only generate the centroid

mtb.vector_tools.sampleExtraction(raster,vector,
                                 outVector=outVector,
                                 uniqueFID='uniquefid',
                                 bandPrefix='band_',
                                 verbose=False)


#############################################
# Read values from both vectors

originalY = mtb.vector_tools.readValuesFromVector(vector,'Class')
extractedY,X = mtb.vector_tools.readValuesFromVector(outVector,'Class',bandPrefix='band_')

#############################################
# Original vector is polygon type, each polygons contains multiple pixel

print(originalY.shape)

#############################################
# Number of Y in the new vector is the total number of pixel in the polygons

print(extractedY.shape)

#############################################
# X has the same size of Y, but in 3 dimensions because our raster has 3 bands

print(X.shape)
print(X[410:420,:])
print(extractedY[410:420])

#############################################
# Plot blue and red band

plt.figure(1)
colors =  [int(i % 23) for i in extractedY]
plt.scatter(X[:,0],X[:,2],c=colors,alpha=.8)
plt.show()

