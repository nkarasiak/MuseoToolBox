# -*- coding: utf-8 -*-
"""
Random Sampling 50% (RS50)
======================================================

This example shows how to make a Random Sampling with 
50% for each class.

"""

##############################################################################
# Import librairies
#^^^^^^^^^^^^^^^^^^^^

from MuseoToolBox.crossValidationTools import RandomCV
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
RS50 = RandomCV(vector,field,train_size=0.5,n_splits=5,
                seed=12,verbose=False)

###############################################################################
# .. note::
#    There is no need to specify a bandPrefix. 
#    If bandPrefix is not specified, scipt will only generate the centroid

for tr,vl in RS50.split():
    print(tr,vl)

