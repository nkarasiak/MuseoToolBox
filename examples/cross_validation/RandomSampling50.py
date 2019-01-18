# -*- coding: utf-8 -*-
"""
Random Sampling 50% (RS50)
======================================================

This example shows how to make a Random Sampling with 
50% for each class.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.cross_validation import RandomCV
from museotoolbox import datasets,raster_tools,vector_tools

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.historicalMap()
field = 'Class'
y = vector_tools.readValuesFromVector(vector,field)

##############################################################################
# Create CV
# -------------------------------------------
RS50 = RandomCV(valid_size=0.5,n_splits=10,
                random_state=12,verbose=False)
for tr,vl in RS50.split(X=None,y=y):
    print(tr,vl)

###############################################################################
# .. note::
#    Split is made to generate each fold

# Show label

for tr,vl in RS50.split(X=None,y=y):
    print(y[tr],y[vl])
    
##############################################################################
# .. note::
#    The first one is made with polygon only.
#    When learning/predicting, all pixels will be taken in account
#    TO generate a full X and y labels, extract samples from ROI

X,y=raster_tools.getSamplesFromROI(raster,vector,field)

for tr,vl in RS50.split(X,y):
    print(tr,vl)
    print(tr.shape,vl.shape)
        
##########################
# Plot example
import numpy as np
from matplotlib import pyplot as plt
plt.scatter(np.random.rand(30),np.random.rand(30),s=100)
plt.scatter(np.random.rand(30),np.random.rand(30),s=100)
plt.axis('off')
plt.show()
