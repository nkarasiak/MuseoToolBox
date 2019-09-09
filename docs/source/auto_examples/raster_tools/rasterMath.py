# -*- coding: utf-8 -*-
"""
Basics to use rasterMath
===============================================================

Compute substract and addition between two raster bands.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.raster_tools import rasterMath
from museotoolbox import datasets
import numpy as np
##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.historicalMap()

##############################################################################
# Initialize rasterMath with raster
# ------------------------------------

rM = rasterMath(raster)

print(rM.getRandomBlock())

##########################
# Let's suppose you want compute the difference between blue and green band
# I suggest you to define type in numpy array to save space while creating the raster!

X = rM.getRandomBlock()
 
sub = lambda X : np.array((X[:,0]-X[:,1])).astype(np.int64) 

####################
# You can use a standard python function too :
#
# def sub(X):
#     return np.array((X[:,0]-X[:,1])).astype(np.int64) 

rM.addFunction(sub,outRaster='/tmp/sub.tif')
#####################
# Run the script

rM.run()

#######################
# Plot result

import gdal
from matplotlib import pyplot as plt 

src = gdal.Open('/tmp/sub.tif')
plt.imshow(src.ReadAsArray())
