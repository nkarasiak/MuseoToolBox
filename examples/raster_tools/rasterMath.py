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

raster,vector = datasets.getHistoricalMap()

##############################################################################
# Initialize rasterMath with raster
# ------------------------------------

rM = rasterMath(raster)

print(rM.getRandomBlock())

##########################
# Let's suppose you want compute the difference between blue and green band
# I suggest you to define type in numpy array to save space while creating the raster!

x = rM.getRandomBlock()

def sub(x):
    return np.array((x[:,0]-x[:,1])).astype(np.int16) 

def add(x):
    
    return np.array((x[:,0]+x[:,1])).astype(np.int16) 

rM.addFunction(sub,outRaster='/tmp/sub.tif')
rM.addFunction(add,outRaster='/tmp/add.tif')

#####################
# Run the script

rM.run()

#######################
# Plot result

import gdal
from matplotlib import pyplot as plt 

src = gdal.Open('/tmp/add.tif')
plt.imshow(src.ReadAsArray())
