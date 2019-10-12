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
# Let's suppose you want compute the difference between blue and green band.
# I suggest you to define type in numpy array to save space while creating the raster!

X = rM.getRandomBlock()

 
sub = lambda X : np.array((X[:,0]-X[:,1])).astype(np.int64) 


rM.addFunction(sub,outRaster='/tmp/sub.tif')

###########################################################
# Use a python function to use arguments
# ------------------------

def sub(X,band1=0,band2=1):
    outX = np.array((X[:,band1]-X[:,band2])).astype(np.int16)
    return outX

#################################################################
# We can add keyword argument in the addFunction.
# This function is going to substract band2 from band 1 

rM = rasterMath(raster)
rM.addFunction(sub,outRaster='/tmp/sub.tif',band1=1,band2=0)

#####################
# Run the script

rM.run()

#######################
# Plot result

import gdal
from matplotlib import pyplot as plt 

src = gdal.Open('/tmp/sub.tif')
plt.imshow(src.ReadAsArray())
