# -*- coding: utf-8 -*-
"""
Stratified-K-Fold
======================================================

This example shows how to make a Random Sampling with 
50% for each class.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.cross_validation import RandomStratifiedKFold
from museotoolbox import datasets,raster_tools,vector_tools

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.load_historical_data(low_res=True)
field = 'Class'
y = vector_tools.read_values(vector,field)

##############################################################################
# Create CV
# -------------------------------------------

SKF = RandomStratifiedKFold(n_splits=2,n_repeats=2,
                random_state=12,verbose=False)
for tr,vl in SKF.split(X=None,y=y):
    print(tr,vl)

###############################################################################
# .. note::
#    Split is made to generate each fold

# Show label

for tr,vl in SKF.split(X=None,y=y):
    print(y[tr],y[vl])

##############################################################################
# .. note::
#    The first one is made with polygon only.
#    When learning/predicting, all pixels will be taken in account
#    TO generate a full X and y labels, extract samples from ROI

X,y=raster_tools.extract_values(raster,vector,field)

for tr,vl in SKF.split(X,y):
    print(tr,vl)
    print(tr.shape,vl.shape)
        
##########################
# Plot example
from __drawCVmethods import plotMethod
plotMethod('SKF-pixel')