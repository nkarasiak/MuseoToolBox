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

raster,vector = mtb.datasets.load_historical_data(low_res=True)

##############################################################################
# Initialize rasterMath with raster
# -----------------------------------------

########
# In case you want to add a mask
mask = '/tmp/maskFromPolygons.tif'

mtb.geo_tools.image_mask_from_vector(vector,raster,out_image = mask)

rM = mtb.geo_tools.RasterMath(raster,in_image_mask=mask)

print(rM.get_random_block())
##########################
# Let's suppose you want compute the modal classification between several predictions
# The first band will be the most predicted class, and the second the number of times it has been predicted.


x = rM.get_random_block()

def modal_class(x):
    tmp = stats.mode(x,axis=1)
    tmpStack = np.column_stack((tmp[0], tmp[1])).astype(np.int16)
    return tmpStack

rM.add_function(modal_class,out_image='/tmp/modal.tif',out_nodata=0)

#####################
# Run the script

rM.run()

#######################
# Plot result

import gdal
from matplotlib import pyplot as plt 

src = gdal.Open('/tmp/modal.tif')
plt.imshow(src.ReadAsArray()[0,:,:])
