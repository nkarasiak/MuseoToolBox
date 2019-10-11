# -*- coding: utf-8 -*-
"""
Spatial Leave-One-SubGroup-Out (SLOSGO)
======================================================

This example shows how to make a Spatial Leave-One-SubGroup-Out.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.cross_validation import SpatialLeaveOneSubGroupOut
from museotoolbox import datasets,raster_tools,vector_tools
##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector,centroid = datasets.historicalMap(low_res=True,centroid=True)

field = 'Class'

##############################################################################
# Extract label ('Class' field) and groups ('uniquefid' field)
# Compute distanceMatrix with centroid (one point per group)

X,y,groups = raster_tools.getSamplesFromROI(raster,vector,field,'uniquefid')
distanceMatrix,distanceLabel = vector_tools.getDistanceMatrix(raster,centroid,'uniquefid')

##############################################################################
# Create CV
# -------------------------------------------
# n_splits will be the number  of the least populated class

SLOSGO = SpatialLeaveOneSubGroupOut(distanceThresold=100,distanceMatrix=distanceMatrix,
                                   distanceLabel=distanceLabel,random_state=12)


###############################################################################
# .. note::
#    Split is made to generate each fold
SLOSGO.get_n_splits(X,y,groups)
for tr,vl in SLOSGO.split(X,y,groups):
    print(tr.shape,vl.shape)
    
#############################################
# Draw image
from __drawCVmethods import plotMethod
plotMethod('SLOO-group')