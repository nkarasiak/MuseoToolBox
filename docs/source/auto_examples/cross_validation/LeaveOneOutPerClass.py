# -*- coding: utf-8 -*-
"""
Leave One Out Per Class (LOOPC)
======================================================

This example shows how to make a Leave One Out for each class.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.cross_validation import LeaveOneOutPerClass
from museotoolbox import datasets

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

X,y = datasets.historicalMap(return_X_y=True)

##############################################################################
# Create CV
# -------------------------------------------
LOOPC = LeaveOneOutPerClass(random_state=8,verbose=False)
for tr,vl in LOOPC.split(X=None,y=y):
    print(tr,vl)

###############################################################################
# .. note::
#    Split is made to generate each fold

# Show label

for tr,vl in LOOPC.split(X=None,y=y):
    print(y[vl])
    
        