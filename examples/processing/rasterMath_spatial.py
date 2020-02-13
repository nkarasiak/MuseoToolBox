# -*- coding: utf-8 -*-
"""
Basics to use rasterMath
===============================================================

Compute substract and addition between two raster bands.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.processing import RasterMath
from museotoolbox import datasets
from numba import jit
import numpy as np
##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------
import time

raster,vector = datasets.load_historical_data()

##############################################################################
# Initialize rasterMath with raster
# ------------------------------------

rM = RasterMath(raster,return_3d=True)

##########################
# Let's suppose you want compute the difference between blue and green band.
# I suggest you to define type in numpy array to save space while creating the raster!

X = rM.get_random_block()

###########################################################
# Use a python function for spatial function
# -------------------------------------------

@jit(nopython=True)
def mean_bands(X):
    res = []
    for x in range(X.shape[-1]):
        res.append(np.mean(X[...,x]))
    return np.asarray(res,dtype=np.int16)

###############################################################################
# You may want to filter an image with a spatial filter (mean filter for instance).
# Just type the offset you want (here 1 means a square of 3x3, 1 as number of neighbors left/up/right/down)


#rM.custom_block_size(256,128)
rM.add_function(mean_bands, out_image='/tmp/mean_bands.tif', offset=1) # set offset to 1
#rM.add_function(mean_bands, out_image='/tmp/mean_bands.tif', offset=3) # set offset to 1
t0 = time.time()
rM.run()
print(time.time() - t0)
#######################
# Plot result

from osgeo import gdal
from matplotlib import pyplot as plt 
src = gdal.Open('/tmp/mean_bands.tif')
plt.figure(figsize=(12,10))
plt.imshow(np.swapaxes(np.swapaxes(src.ReadAsArray().astype(np.int16),0,2).astype(np.int16),0,1))