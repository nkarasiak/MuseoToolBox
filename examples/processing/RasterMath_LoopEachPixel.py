#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Using rasterMath with 3d block or 2d block
==================================================================

Test notebook to validate code.
"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.processing import RasterMath,image_mask_from_vector
from museotoolbox import datasets
from matplotlib import pyplot as plt
import numpy as np

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.load_historical_data()

##############################################################################
# Initialize rasterMath with raster
# ------------------------------------

# Set return_3d to True to have full block size (not one pixel per row)
# Create raster mask to only keep pixel inside polygons.

image_mask_from_vector(vector,raster,'/tmp/mask.tif',invert=True)

import time

from numba import jit

@jit(nopython=True)
def loop_in_image(X):
    x = np.full(X.shape,5)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j,...] = X[i,j,...]
    
    return x


rM = RasterMath(raster,in_image_mask='/tmp/mask.tif',return_3d=True,verbose=0)
from joblib import Parallel,delayed

t0 = time.time()
Parallel(-1)(delayed(loop_in_image)(rM.get_block(X)) for X in range(3))
print(time.time()-t0)

t0 = time.time()
for X in range(3):
    loop_in_image(rM.get_block(X))
print(time.time()-t0)
#    print(rM.get_random_block().shape)

X = rM.get_random_block()

# Returns with only 1 dimension
returnFlatten = lambda x : x[...,0]

# Returns 3x the original last dimension
# Add functions to rasterMath
rM.add_function(loop_in_image,'/tmp/loop.tif')
t=time.time()

rM.run()
#rM.run()

print(time.time()-t)
from osgeo import gdal
dst = gdal.Open('/tmp/loop.tif')
arr = dst.GetRasterBand(1).ReadAsArray()
plt.imshow(np.ma.masked_where(arr == np.min(arr), arr))
