# -*- coding: utf-8 -*-
"""
rasterMath with custom block size, mask, and in 3 dimensions
===================================================================

Tips to use rasterMath by defining its block size and to receive
a full block (not a array with one pixel per row.)

Tips : A function readBlockPerBlock() yields each block, without saving results
to a new raster.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.raster_tools import rasterMath,rasterMaskFromVector
from museotoolbox import datasets
from matplotlib import pyplot as plt
import numpy as np
##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.historicalMap()

##############################################################################
# Initialize rasterMath with raster
# ------------------------------------

# Set return_3d to True to have full block size (not one pixel per row)
# Create raster mask to only keep pixel inside polygons.

rasterMaskFromVector(vector,raster,'/tmp/mask.tif',invert=False)

rM = rasterMath(raster,inMaskRaster='/tmp/mask.tif',return_3d=True)
#rM.addInputRaster('/tmp/mask.tif')
print(rM.getRandomBlock()[0].shape)

#######################
# Plot blocks
x = rM.getRandomBlock()

rM.addFunction(np.mean,'/tmp/mean.tif',axis=2,dtype=np.int16)

for tile in rM.readBlockPerBlock():
    print(tile)
#rM.addFunction(returnX,'/tmp/mean.tif')
rM.run()

import gdal
dst = gdal.Open('/tmp/mean.tif')
arr = dst.GetRasterBand(1).ReadAsArray()
plt.imshow(np.ma.masked_where(arr == np.min(arr), arr))