# -*- coding: utf-8 -*-
"""
Spatial Leave-Aside-Out (SLAO)
======================================================

This example shows how to make a Spatial Leave-Aside-Out.

See https://doi.org/10.1016/j.foreco.2013.07.059

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.cross_validation import SpatialLeaveAsideOut
from museotoolbox import datasets,raster_tools,vector_tools

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.getHistoricalMap()
field = 'Class'
X,y = raster_tools.getSamplesFromROI(raster,vector,field)
distanceMatrix = vector_tools.getDistanceMatrix(raster,vector)

##############################################################################
# Create CV
# -------------------------------------------
# n_splits will be the number  of the least populated class

SLOPO = SpatialLeaveAsideOut(valid_size=0.5,n_splits=2,
                             distanceMatrix=distanceMatrix,random_state=12)

print(SLOPO.get_n_splits(X,y))

###############################################################################
# .. note::
#    Split is made to generate each fold

for tr,vl in SLOPO.split(X,y):
    print(tr.shape,vl.shape)  

###############################################################################
#    Save each train/valid fold in a file
# -------------------------------------------
# In order to translate polygons into points (each points is a pixel in the raster)
# we use sampleExtraction from vector_tools to generate a temporary vector.

vector_tools.sampleExtraction(raster,vector,outVector='/tmp/pixels.gpkg',verbose=False)
trvl = SLOPO.saveVectorFiles('/tmp/pixels.gpkg',field,outVector='/tmp/SLOPO.gpkg')
for tr,vl in trvl:
    print(tr,vl)