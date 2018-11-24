# -*- coding: utf-8 -*-
"""
Leave-One-SubGroup-Out (LOSGO)
======================================================

This example shows how to make a Leave-One-SubGroup-Out.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from MuseoToolBox.crossValidationTools import LeaveOneSubGroupOut
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

valid_size = 0.5 # Means 50%
LOSGO = LeaveOneSubGroupOut(vector,field,group,
                            verbose=False,random_state=12)

###############################################################################
# .. note::
#    There is no need to specify a bandPrefix. 
#    If bandPrefix is not specified, scipt will only generate the centroid

for tr,vl in LOSGO.split():
    print(tr,vl)

###############################################################################
# Differences with sklearn
# -------------------------------------------
# Sklearn do not use subgroups
# as MuseoToolBox use one group per Y label    
from sklearn.model_selection import LeaveOneGroupOut
from MuseoToolBox import vectorTools

Y,Groups = vectorTools.readValuesFromVector(vector,field,group)
LOGO = LeaveOneGroupOut()
for tr,vl in LOGO.split(X=Y,y=Y,groups=Groups):
    print(tr,vl)
# Plot example in image
    
import numpy as np
from matplotlib import pyplot as plt
plt.scatter(np.random.randint(10,30,40),np.random.randint(10,30,40),s=100,color='#1f77b4')
plt.scatter(np.random.randint(0,10,40),np.random.randint(10,30,40),s=100,color='#1f77b4')
plt.scatter(np.random.randint(0,10,20),np.random.randint(0,10,20),s=100,color='#ff7f0e')
plt.axis('off')
plt.show()