# -*- coding: utf-8 -*-
"""
rasterMath with several rasters as inputs
===============================================================

Compute substract and addition between two raster bands.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.raster_tools import rasterMath,rasterMaskFromVector
from museotoolbox import datasets
import numpy as np
##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.historicalMap()

##############################################################################
# Initialize rasterMath with raster
# ------------------------------------

##############################################################################
# If invert is set to True, it means polygons will be set to nodata

rasterMaskFromVector(vector,raster,'/tmp/mask.tif',invert=True)
rM = rasterMath(raster,inMaskRaster='/tmp/mask.tif',return_3d=False)
rM.addInputRaster(raster)

print(rM.getRandomBlock())

##########################
# Let's suppose you want compute the add of the blue and green band of two inputs
# I suggest you to define type in numpy array to save space while creating the raster!

x = rM.getRandomBlock()

def add(x):
    return np.array((x[0][...,0]-x[1][...,1])).astype(np.uint8)

rM.addFunction(add,outRaster='/tmp/add.tif',outNumpyDT=np.uint8)

#####################
# Run the script

rM.run()

#######################
# Plot result

import gdal
from matplotlib import pyplot as plt 

src = gdal.Open('/tmp/add.tif')
arr = src.ReadAsArray()
arr = np.where(arr==0,np.nan,arr)
plt.imshow(arr)
