# -*- coding: utf-8 -*-
"""
Extract raster values from vector file
===============================================================

Easily extract raster values from vector files (polygon/point)
"""

##############################################################################
# Import librairies
# -------------------------------------------

import museotoolbox as mtb
import numpy as np
##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = mtb.datasets.load_historical_data() 

##############################################################################
# Extract raster values with no vector information
# -------------------------------------------------

X = mtb.geo_tools.extract_ROI(raster,vector)

print("Vector contains {} pixels".format(X.shape[0]))
print("Raster contains {} bands per pixel".format(X.shape[1]))

##########################
# Let's suppose you want konw to extract the label of each polygon/point

X,y = mtb.geo_tools.extract_ROI(raster,vector,'class')
uniqueLabels = np.unique(y,return_counts=True)

for label,count in zip(*uniqueLabels):
    print('label {} has {} samples'.format(label,count))
    
####################
# You can put as many fields as you want, except fields of string type

X,y,g = mtb.geo_tools.extract_ROI(raster,vector,'class','uniquefid')
print('There are a total of {} groups'.format(np.unique(g).size))