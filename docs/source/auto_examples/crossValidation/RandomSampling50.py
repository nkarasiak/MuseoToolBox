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

from MuseoToolBox.crossValidationTools import RandomCV
from MuseoToolBox import datasets

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.getHistoricalMap()
field = 'Class'
group = 'uniquefid'

##############################################################################
# Create CV
# -------------------------------------------
RS50 = RandomCV(vector,field,train_size=0.5,n_splits=10,
                random_state=12,verbose=False)

for tr,vl in RS50.split():
    print(tr,vl)
    
######################################
# Show label of eahc train/validation
from MuseoToolBox import vectorTools
Y=vectorTools.readValuesFromVector(vector,field)
for tr,vl in RS50.split():
    print(Y[tr],Y[vl])

##############################################################################
# .. note::
#    The first one is made with polygon id only.
#    When learning/predicting, all pixels will be taken in account

from MuseoToolBox import rasterTools
X,Y = rasterTools.getSamplesFromROI(raster,vector,field)

for tr,vl in RS50.split(X,Y):
    print(tr.shape,vl.shape)
    
    
##########################
# Plot example
import numpy as np
from matplotlib import pyplot as plt
plt.scatter(np.random.rand(30),np.random.rand(30),s=100)
plt.scatter(np.random.rand(30),np.random.rand(30),s=100)
plt.axis('off')
plt.show()
