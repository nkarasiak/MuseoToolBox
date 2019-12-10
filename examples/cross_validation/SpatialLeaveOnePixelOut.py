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
from museotoolbox import datasets,processing
##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.load_historical_data(low_res=True)
field = 'Class'
X,y = processing.extract_ROI(raster,vector,field)
distance_matrix = processing.get_distance_matrix(raster,vector)

##############################################################################
# Create CV
# -------------------------------------------
# n_splits will be the number  of the least populated class

SLOO = SpatialLeaveOneOut(distance_thresold=100,distance_matrix=distance_matrix,
                                random_state=12)
###############################################################################
# .. note::
#    Split is made to generate each fold
SLOO.get_n_splits(X,y)
for tr,vl in SLOO.split(X,y):
    print(tr.shape,vl.shape)

####################################################
# Save each train/valid in a spatial vector file
from museotoolbox.processing import sample_extraction
sample_extraction(raster,vector,'/tmp/one_point_per_pixel.gpkg')
files = SLOO.save_to_vector('/tmp/one_point_per_pixel.gpkg','Class',out_vector='/tmp/trvl.gpkg')
print(files)
#############################################
# Draw image
from __drawCVmethods import plotMethod
plotMethod('SLOO-pixel')
