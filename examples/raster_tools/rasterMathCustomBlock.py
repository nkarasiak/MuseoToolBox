# -*- coding: utf-8 -*-
"""
rasterMath with custom window/block size (and with 3 dimensions)
=================================================================

Tips to use rasterMath by defining its block size and to receive
a full block (not a array with one pixel per row.)

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.raster_tools import rasterMath
from museotoolbox import datasets
from matplotlib import pyplot as plt

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.historicalMap()

##############################################################################
# Initialize rasterMath with raster
# ------------------------------------

# Set return3d to True to have full block size (not one pixel per row)

rM = rasterMath(raster,return_3d=True)

print(rM.getRandomBlock().shape)

##############################################################################
# Comparing different block size (%, fixed, full block)
# -------------------------------------------------------

####################### 
# You can define block by percentage of the whole width/height

rM.customBlockSize(1/2,1/2) 
print(rM.getRandomBlock().shape)

#######################
# Or by fixed window 

rM.customBlockSize(50,100) # width divided every 50 pixel and height every 100
print(rM.getRandomBlock().shape)

########################
# To have the full image (one block)

rM.customBlockSize(-1,-1) # to have the full image

########################
# To have block width divided by 4 and height by 2

rM.customBlockSize(1/4,1/2)

##########################################
# Define block size for output raster
# -------------------------------------

raster_parameters = rM.getRasterParameters()

print('Default parameters are '+str(raster_parameters))


# to do before adding the function

rM.customBlockSize(256,256) # custom for reading AND writing the output
#raster_parameters = ['COMPRESS=DEFLATE']
#rM.customRasterParameters(raster_parameters)

#####################################
# now add a function to just return the same raster

returnSameImage  = lambda x : x
rM.addFunction(returnSameImage,'/tmp/testcustomblock.tif')
rM.run()

#####################
# check block size of new raster

rMblock = rasterMath('/tmp/testcustomblock.tif')
print(rMblock.block_sizes)

#######################
# Plot blocks

n_row,n_col = 2,4
rM.customBlockSize(1/n_col,1/n_row)

fig=plt.figure(figsize=(12,6),dpi=150)

for idx,tile in enumerate(rM.readBlockPerBlock()):
    fig.add_subplot(n_row,n_col,idx+1)
    plt.title('block %s' %(idx+1))
    plt.imshow(tile)
plt.show()

