# -*- coding: utf-8 -*-
"""
Generate a cross-validation and/or save each fold to a vector file
===================================================================

This example shows how to make a Leave-One-SubGroup-Out and save
each fold as a vector file.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.cross_validation import LeaveOneSubGroupOut
from museotoolbox.processing import extract_ROI
from museotoolbox import datasets

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.load_historical_data(low_res=True)
field = 'Class'
group = 'uniquefid'
X,y,s = extract_ROI(raster,vector,field,group)
##############################################################################
# Create CV
# -------------------------------------------

valid_size = 0.5 # Means 50%
LOSGO = LeaveOneSubGroupOut(verbose=False,random_state=12)

###############################################################################
# .. note::
#    Split is made to generate each fold

LOSGO.get_n_splits(X,y,s)
for tr,vl in LOSGO.split(X,y,s):
    print(tr.shape,vl.shape)

###############################################################################
#  Save each train/valid fold to a vector file (here in polygon type)
#

vectorFiles = LOSGO.save_to_vector(vector,field,group=group,out_vector='/tmp/LOSGO.gpkg')

for tr,vl in vectorFiles:
    print(tr,vl)

###############################################################################
#  The sampling can be different in vector point or polygon.
#  So you can generate each centroid of a pixel that contains the polygon.
# 
    
from museotoolbox.processing import sample_extraction
vectorPointPerPixel = '/tmp/vectorCentroid.gpkg'
sample_extraction(raster,vector,vectorPointPerPixel)

vectorFiles = LOSGO.save_to_vector(vectorPointPerPixel,field,group=group,out_vector='/tmp/LOSGO.gpkg')

for tr,vl in LOSGO.split(X,y,s):
    print(tr.shape,vl.shape)