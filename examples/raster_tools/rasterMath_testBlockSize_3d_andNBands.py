# -*- coding: utf-8 -*-
"""
Tests rasterMath with full block or stacken and custom block size
==================================================================

Test notebook in order to validate code.

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

for return_3d in [False,True]:
    rM = rasterMath(raster,inMaskRaster='/tmp/mask.tif',return_3d=return_3d)
    
    rM.customBlockSize(200,200) # block of 200x200pixels
    
    print(rM.getRandomBlock().shape)
    
    #######################
    # Plot blocks
    x = rM.getRandomBlock()
    def returnFlatten(x):
        try:
            x = x[:,:,0]
        except:
            x = x[:,0].reshape(-1,1)
        return x
    def returnWithOneBandMore(x):
        try:
            x = np.repeat(x,3,axis=2)
        except:
            x= np.repeat(x,3,axis=1)
        return x
    
    rM.addFunction(returnWithOneBandMore,'/tmp/x_repeat_{}.tif'.format(str(return_3d)))
    rM.addFunction(returnFlatten,'/tmp/x_flatten_{}.tif'.format(str(return_3d)))

    
    
    rM.run()
    
import gdal
dst = gdal.Open('/tmp/x_flatten_False.tif')
arr = dst.GetRasterBand(1).ReadAsArray()
plt.imshow(np.ma.masked_where(arr == np.min(arr), arr))
