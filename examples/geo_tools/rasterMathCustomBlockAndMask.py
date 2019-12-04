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

from museotoolbox.geo_tools import RasterMath,image_mask_from_vector
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

image_mask_from_vector(vector,raster,'/tmp/mask.tif',invert=False)

rM = RasterMath(raster,in_image_mask='/tmp/mask.tif',return_3d=True)
#rM.addInputRaster('/tmp/mask.tif')
print(rM.get_random_block().shape)

#######################
# Plot blocks
x = rM.get_random_block()

rM.add_function(np.mean,'/tmp/mean.tif',axis=2,out_np_dt=np.int16)

rM.run()

from osgeo import gdal
dst = gdal.Open('/tmp/mean.tif')
arr = dst.GetRasterBand(1).ReadAsArray()
plt.imshow(np.ma.masked_where(arr == rM.outputNoData[0], arr))
