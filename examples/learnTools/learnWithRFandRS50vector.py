# -*- coding: utf-8 -*-
"""
Learn from vector with Random-Forest and Random Sampling 50% (RS50)
====================================================================

This example shows how to make a Random Sampling with 
50% for each class.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.learnTools import learnAndPredict
from museotoolbox.crossValidation import RandomCV
from museotoolbox import datasets
from sklearn.ensemble import RandomForestClassifier

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

X,y = datasets.getHistoricalMap(return_X_y=True)

##############################################################################
# Create CV
# -------------------------------------------
RS50 = RandomCV(valid_size=0.5,n_splits=10,
                random_state=12,verbose=False)

##############################################################################
# Initialize Random-Forest
# ---------------------------

classifier = RandomForestClassifier(random_state=12)

##############################################################################
# Start learning
# ---------------------------

LAP = learnAndPredict(n_jobs=-1)
LAP.learnFromVector(X,y,cv=RS50,
                    classifier=classifier,param_grid=dict(n_estimators=[100,200]))

##############################################################################
# Get kappa from each fold
# ---------------------------
  
for kappa in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
    print(kappa)

##############################################################################
# Get each confusion matrix from folds
# -----------------------------------------------

for cm in LAP.getStatsFromCV(confusionMatrix=True):
    print(cm)
    
##############################################################################
# Save each confusion matrix from folds
# -----------------------------------------------

LAP.saveCMFromCV('/tmp/testMTB/',prefix='RS50_',header=False)

##############################################################################
# Predict map
# ---------------------------
raster,_ = datasets.getHistoricalMap()
LAP.predictRaster(raster,'/tmp/classification.tif')

##########################
# Plot example

from matplotlib import pyplot as plt
import gdal
src=gdal.Open('/tmp/classification.tif')
plt.imshow(src.GetRasterBand(1).ReadAsArray(),cmap=plt.get_cmap('tab20'))
plt.axis('off')
plt.show()
