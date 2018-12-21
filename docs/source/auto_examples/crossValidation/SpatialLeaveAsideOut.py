# -*- coding: utf-8 -*-
"""
Spatial Leave-Aside-Out (SLAO)
======================================================

This example shows how to make a Spatial Leave-Aside-Out.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.crossValidation import SpatialLeaveAsideOut
from museotoolbox import datasets,rasterTools,vectorTools
##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.getHistoricalMap()
field = 'Class'
X,y = rasterTools.getSamplesFromROI(raster,vector,field)
distanceMatrix = vectorTools.getDistanceMatrix(raster,vector)

##############################################################################
# Create CV
# -------------------------------------------
# n_splits will be the number  of the least populated class

SLOPO = SpatialLeaveAsideOut(valid_size=0.5,n_splits=10,
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
# we use sampleExtraction from vectorTools to generate a temporary vector.

vectorTools.sampleExtraction(raster,vector,outVector='/tmp/pixels.gpkg')

SLOPO.saveVectorFiles('/tmp/pixels.gpkg',field,outVector='/tmp/SLOPO.gpkg')