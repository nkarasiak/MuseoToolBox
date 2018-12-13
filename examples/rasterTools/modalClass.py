# -*- coding: utf-8 -*-
"""
Modal class and number of agreements
===============================================================

Create a raster with the modal class and the number of agreements.

"""

##############################################################################
# Import librairies
# -------------------------------------------

import museotoolbox as mtb
from scipy import stats
import numpy as np
##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = mtb.datasets.getHistoricalMap()

##############################################################################
# Initialize rasterMath with raster
# -----------------------------------------

rM = mtb.rasterTools.rasterMath(raster)

print(rM.getRandomBlock())

##########################
# Let's suppose you want compute the modal classification between several predictions
# The first band will be the most predicted class, and the second the number of times it has been predicted.


x = rM.getRandomBlock()

def modalClass(x):
    tmp = stats.mode(x, axis=-1)
    tmpStack = np.column_stack((tmp[0], tmp[1])).astype(np.int16)
    return tmpStack

rM.addFunction(modalClass,outRaster='/tmp/modal.tif')

#####################
# Run the script

rM.run()

#######################
# Plot result

import gdal
from matplotlib import pyplot as plt 

src = gdal.Open('/tmp/modal.tif')
plt.imshow(src.ReadAsArray()[0,:,:])
