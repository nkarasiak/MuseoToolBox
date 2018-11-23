# -*- coding: utf-8 -*-
"""
Spatial Leave-One-Pixel-Out / No raster (SLOPO)
======================================================

This example shows how to make a Spatial Leave-One-Out called here
a Spatial Leave-One-Pixel-Out.

For more information see : https://onlinelibrary.wiley.com/doi/full/10.1111/geb.12161.

"""

##############################################################################
# Import librairies
#^^^^^^^^^^^^^^^^^^^^

from MuseoToolBox.crossValidationTools import SpatialLeaveOnePixelOut
from MuseoToolBox.vectorTools import getDistanceMatrix
from MuseoToolBox.rasterTools import getSamplesFromROI
from MuseoToolBox import datasets
##############################################################################
# Load HistoricalMap dataset
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

raster,vector = datasets.getHistoricalMap()
field = 'Class'

X,Y = getSamplesFromROI(raster,vector,field,verbose=False)
distanceMatrix = getDistanceMatrix(raster,vector,verbose=False)

##############################################################################
# Create CV
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SLOPO = SpatialLeaveOnePixelOut(None,Y,None,distanceMatrix=distanceMatrix,distanceThresold=100,seed=12,verbose=False)

###############################################################################
# .. note::
#    There is no need to specify a bandPrefix. 
#    If bandPrefix is not specified, scipt will only generate the centroid

for tr,vl in SLOPO.split():
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
