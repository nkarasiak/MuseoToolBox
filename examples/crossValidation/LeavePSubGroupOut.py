# -*- coding: utf-8 -*-
"""
Leave-P-SubGroup-Out (LPSGO)
======================================================

This example shows how to make a Leave-Percent-SubGroup-Out.

"""

##############################################################################
# Import librairies
#^^^^^^^^^^^^^^^^^^^^

from MuseoToolBox.crossValidationTools import LeavePSubGroupOut
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
LPSGO = LeavePSubGroupOut(vector,field,group,
                          valid_size = valid_size,n_splits = 5,
                          seed=12,verbose=False)

###############################################################################
# .. note::
#    There is no need to specify a bandPrefix. 
#    If bandPrefix is not specified, scipt will only generate the centroid

for tr,vl in LPSGO.split():
    print(tr,vl)

