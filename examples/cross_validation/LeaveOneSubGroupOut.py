# -*- coding: utf-8 -*-
"""
Leave-One-SubGroup-Out (LOSGO)
======================================================

This example shows how to make a Leave-One-SubGroup-Out.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.cross_validation import LeaveOneSubGroupOut
from museotoolbox.raster_tools import getSamplesFromROI
from museotoolbox import datasets

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.getHistoricalMap()
field = 'Class'
group = 'uniquefid'
X,y,s = getSamplesFromROI(raster,vector,field,group)

##############################################################################
# Create CV
# -------------------------------------------
# if n_splits is False (default), the number of splits will be the smallest
# number of subgroup of all labels.

valid_size = 0.5 # Means 50%
LOSGO = LeaveOneSubGroupOut(verbose=False,random_state=12) # 

###############################################################################
# .. note::
#    Split is made to generate each fold

LOSGO.get_n_splits(X,y,s)
for tr,vl in LOSGO.split(X,y,s):
    print(tr.shape,vl.shape)

###############################################################################
# Differences with sklearn
# ------------------------------------------- 
# Sklearn do not use subgroups (only groups), so no hierarchical dependances.
    
from sklearn.model_selection import LeaveOneGroupOut
LOGO = LeaveOneGroupOut()
for tr,vl in LOGO.split(X=X,y=y,groups=s):
    print(tr.shape,vl.shape)

###############################################################################
# Plot example
    
import numpy as np
from matplotlib import pyplot as plt
plt.scatter(np.random.randint(10,30,40),np.random.randint(10,30,40),s=100,color='#1f77b4')
plt.scatter(np.random.randint(0,10,40),np.random.randint(10,30,40),s=100,color='#1f77b4')
plt.scatter(np.random.randint(0,10,20),np.random.randint(0,10,20),s=100,color='#ff7f0e')
plt.axis('off')
plt.show()