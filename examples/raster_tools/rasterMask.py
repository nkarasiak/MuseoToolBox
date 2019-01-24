# -*- coding: utf-8 -*-
"""
Raster mask from vector
===============================================================

Create a raster mask from vector.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox import raster_tools
from museotoolbox.datasets import historicalMap

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = historicalMap()

##############################################################################
# Rasterize vector
# -----------------------------------------


##############################################################################
# Default, invert is False, it means only polygons will be kept (the rest is set to nodata)
raster_tools.rasterMaskFromVector(vector,raster,'/tmp/mask.tif',invert=False)

##############################################################################
# If invert is set to True, it means polygons will be set to nodata
raster_tools.rasterMaskFromVector(vector,raster,'/tmp/maskInvert.tif',invert=True)

#######################
# Plot result
# -----------------------------------

import gdal
from matplotlib import pyplot as plt 

##############################################
# Default mask (invert=False)
# -----------------------------------
# white is nodata, black is 255
src = gdal.Open('/tmp/mask.tif')
plt.imshow(src.ReadAsArray(),cmap='Greys')

##############################################
# invert mask (invert=True)
# -----------------------------------
# white is nodata, black is 255
src = gdal.Open('/tmp/maskInvert.tif')
plt.imshow(src.ReadAsArray(),cmap='Greys')
