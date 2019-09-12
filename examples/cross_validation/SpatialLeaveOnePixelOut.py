# -*- coding: utf-8 -*-
"""
Spatial Leave-One-Out (SLOO)
======================================================

This example shows how to make a Spatial Leave-One-Out called here
a Spatial Leave-One-Pixel-Out.

For more information see : https://onlinelibrary.wiley.com/doi/full/10.1111/geb.12161.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.cross_validation import SpatialLeaveOneOut
from museotoolbox import datasets,raster_tools,vector_tools
##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.historicalMap(low_res=True)
field = 'Class'
X,y = raster_tools.getSamplesFromROI(raster,vector,field)
distanceMatrix = vector_tools.getDistanceMatrix(raster,vector)

##############################################################################
# Create CV
# -------------------------------------------
# n_splits will be the number  of the least populated class

SLOPO = SpatialLeaveOneOut(distanceThresold=100,distanceMatrix=distanceMatrix,
                                random_state=12)


print(SLOPO.get_n_splits(X,y))


###############################################################################
# .. note::
#    Split is made to generate each fold

for tr,vl in SLOPO.split(X,y):
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
