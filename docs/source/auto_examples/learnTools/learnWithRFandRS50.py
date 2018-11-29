# -*- coding: utf-8 -*-
"""
Learn with Random-Forest and Random Sampling 50% (RS50)
========================================================

This example shows how to make a Random Sampling with 
50% for each class.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from MuseoToolBox.crossValidationTools import RandomCV
from MuseoToolBox import datasets,rasterTools,vectorTools
from MuseoToolBox import learnTools
from sklearn.ensemble import RandomForestClassifier

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.getHistoricalMap()
field = 'Class'
##############################################################################
# Create CV
# -------------------------------------------
RS50 = RandomCV(train_size=0.5,n_splits=10,
                random_state=12,verbose=False)

##############################################################################
# Initialize Random-Forest
# ---------------------------

classifier = RandomForestClassifier()

##############################################################################
# Initialize Random-Forest
# ---------------------------


LAP = learnTools.learnAndPredict()
LAP.learnFromRaster(raster,vector,field,cv=RS50,
                    classifier=classifier,param_grid=dict(n_estimators=[100,200]))
  
for kappa in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
    print(kappa)

##############################################################################
# Initialize Random-Forest
# ---------------------------
    
LAP.predictFromRaster(raster,'/tmp/classification.tif')

##########################
# Plot example


import numpy as np
from matplotlib import pyplot as plt
import gdal
src=gdal.Open('/tmp/classification.tif')
plt.imshow(src.GetRasterBand(1).ReadAsArray(),cmap=plt.get_cmap('tab20'))
plt.axis('off')
plt.show()
