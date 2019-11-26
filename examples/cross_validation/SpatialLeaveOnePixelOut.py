# -*- coding: utf-8 -*-
"""
Spatial Leave-One-Out (SLOO)
======================================================

This example shows how to make a Spatial Leave-One-Out called here
a Spatial Leave-One-Pixel-Out.

For more information see : https://onlinelibrary.wiley.com/doi/full/10.1111/geb.12161.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.cross_validation import SpatialLeaveOneOut
from museotoolbox import datasets,geo_tools
##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.load_historical_data(low_res=True)
field = 'Class'
X,y = geo_tools.extract_ROI(raster,vector,field)
distance_matrix = geo_tools.get_distance_matrix(raster,vector)

##############################################################################
# Create CV
# -------------------------------------------
# n_splits will be the number  of the least populated class

SLOPO = SpatialLeaveOneOut(distance_thresold=100,distance_matrix=distance_matrix,
                                random_state=12)


print(SLOPO.get_n_splits(X,y))


###############################################################################
# .. note::
#    Split is made to generate each fold

for tr,vl in SLOPO.split(X,y):
    print(tr.shape,vl.shape)

#############################################
# Draw image
from __drawCVmethods import plotMethod
plotMethod('SLOO-pixel')
