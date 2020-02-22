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

for return_3d in [True,False]:

    rM = RasterMath(raster,in_image_mask='/tmp/mask.tif',return_3d=return_3d)
    
    rM.custom_block_size(128,128) # block of  128x128
    
    x = rM.get_block()
    
    # Returns with only 1 dimension
    returnFlatten = lambda x : x[...,0]
    
    # Returns 3x the original last dimension
    addOneBand = lambda x : np.repeat(x,3,axis=x.ndim-1)
    # Add functions to rasterMath
    rM.add_function(addOneBand,'/tmp/x_repeat_{}.tif'.format(str(return_3d)))
    rM.add_function(returnFlatten,'/tmp/x_flatten_{}.tif'.format(str(return_3d)))
    t=time.time()
    
    rM.run()
    
    print(time.time()-t)

from osgeo import gdal
dst = gdal.Open('/tmp/x_flatten_True.tif')
arr = dst.GetRasterBand(1).ReadAsArray()
plt.imshow(np.ma.masked_where(arr == np.min(arr), arr))
