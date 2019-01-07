# -*- coding: utf-8 -*-
"""
Leave One Pixel Out Per Class (LOPOPC)
======================================================

This example shows how to make a Leave One Pixel Out for each class.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.crossValidation import RandomCV
from museotoolbox import datasets,rasterTools,vectorTools

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

X,y = datasets.getHistoricalMap(return_X_y=True)

##############################################################################
# Create CV
# -------------------------------------------
RS50 = RandomCV(valid_size=1,n_splits=np.amin(np.unique(y,return_counts=True)[1]),
                random_state=8,verbose=False)
for tr,vl in RS50.split(X=None,y=y):
    print(tr,vl)

###############################################################################
# .. note::
#    Split is made to generate each fold

# Show label

for tr,vl in RS50.split(X=None,y=y):
    print(y[tr],y[vl])
    
        