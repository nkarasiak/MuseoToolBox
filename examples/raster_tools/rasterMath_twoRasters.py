# -*- coding: utf-8 -*-
"""
rasterMath with several rasters as inputs
===============================================================

Compute substract and addition between two raster bands.
"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.raster_tools import RasterMath,image_mask_from_vector
from museotoolbox import datasets
import numpy as np
##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.load_historical_data()

##############################################################################
# Initialize rasterMath with raster
# ------------------------------------

##############################################################################
# If invert is set to True, it means polygons will be set to nodata

image_mask_from_vector(vector,raster,'/tmp/mask.tif',invert=True)
rM = RasterMath(in_image = raster,in_image_mask='/tmp/mask.tif',return_3d=False)

rM.add_image(raster)

print('Number of rasters : '+str(len(rM.get_random_block())))

##########################
# Let's suppose you want compute the substraction between the blue and green band of two inputs
# I suggest you to define type in numpy array to save space while creating the raster!

x = rM.get_random_block()

def sub(x):
    firstBandOfFirstRaster = x[0][...,0]
    thirdBandOfSecondRaster = x[1][...,2]
    difference = np.array(firstBandOfFirstRaster-thirdBandOfSecondRaster,dtype=np.uint8)
    return difference

rM.add_function(sub,out_image='/tmp/sub_2inputs.tif')

#####################
# Run the script

rM.run()

#######################
# Plot result

import gdal
from matplotlib import pyplot as plt 

src = gdal.Open('/tmp/sub_2inputs.tif')
arr = src.ReadAsArray()
arr = np.where(arr==0,np.nan,arr)
plt.imshow(arr)
