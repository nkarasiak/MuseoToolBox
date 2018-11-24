# -*- coding: utf-8 -*-
"""
Leave-One-SubGroup-Out (LOSGO)
======================================================

This example shows how to make a Leave-One-SubGroup-Out.

"""

##############################################################################
# Import librairies
#^^^^^^^^^^^^^^^^^^^^

from MuseoToolBox.crossValidationTools import LeaveOneSubGroupOut
from MuseoToolBox import datasets

##############################################################################
# Load HistoricalMap dataset
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

raster,vector = datasets.getHistoricalMap()
field = 'Class'
group = 'uniquefid'

##############################################################################
# Create CV
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
valid_size = 0.5 # Means 50%
LPSGO = LeaveOneSubGroupOut(vector,field,group,n_splits = 10,verbose=False)

###############################################################################
# .. note::
#    There is no need to specify a bandPrefix. 
#    If bandPrefix is not specified, scipt will only generate the centroid

for tr,vl in LPSGO.split():
    print(tr,vl)

###############################################################################
# Plot example in image
    
import numpy as np
from matplotlib import pyplot as plt
plt.scatter(np.random.randint(10,30,40),np.random.randint(10,30,40),s=100,color='#1f77b4')
plt.scatter(np.random.randint(0,10,40),np.random.randint(10,30,40),s=100,color='#1f77b4')
plt.scatter(np.random.randint(0,10,20),np.random.randint(0,10,20),s=100,color='#ff7f0e')
plt.axis('off')
plt.show()