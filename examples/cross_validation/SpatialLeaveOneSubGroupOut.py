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

raster,vector,centroid = datasets.getHistoricalMap(centroid=True)

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

for tr,vl in SLOSGO.split(X,y,groups):
    print(tr.shape,vl.shape)
    
#############################################
# Draw image
import numpy as np
from matplotlib import pyplot as plt
fig, ax = plt.subplots()
plt.ylim(40,150)
plt.xlim(40,150)


plt.scatter(np.random.randint(50,150,50),np.random.randint(50,150,50),alpha=.8)
plt.scatter(80,80, s=80*100,alpha=.8)
plt.scatter(80,80,color='green',s=60)
plt.text(82,82,'Validation pixel',size=12)
plt.text(110,110,'Training pixels',size=12)
plt.text(46,52,'Buffer of spatial auto-correlated pixels')
plt.axis('off')

plt.show()

