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

from museotoolbox.learn_tools import learnAndPredict
from museotoolbox.cross_validation import RandomStratifiedKFold
from museotoolbox import datasets
from sklearn.ensemble import RandomForestClassifier

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

X,y = datasets.historicalMap(return_X_y=True,low_res=True)

##############################################################################
# Create CV
# -------------------------------------------
SKF = RandomStratifiedKFold(n_splits=2,n_repeats=5,
                random_state=12,verbose=False)

##############################################################################
# Initialize Random-Forest
# ---------------------------

classifier = RandomForestClassifier(random_state=12)

##############################################################################
# Start learning
# ---------------------------

LAP = learnAndPredict(n_jobs=1)
LAP.learnFromVector(X,y,cv=SKF,
                    classifier=classifier,param_grid=dict(n_estimators=[10]))

##############################################################################
# Get kappa from each fold
# ---------------------------
  
for stats in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
    print(stats['kappa'])

##############################################################################
# Get each confusion matrix from folds
# -----------------------------------------------

for stats in LAP.getStatsFromCV(confusionMatrix=True):
    print(stats['confusionMatrix'])
    
##############################################################################
# Only get accuracies score (OA and Kappa)
# -----------------------------------------------

for stats in LAP.getStatsFromCV(OA=True,kappa=True,confusionMatrix=False,F1=False):
    print(stats)
    
##############################################################################
# Save each confusion matrix from folds
# -----------------------------------------------

LAP.saveCMFromCV('/tmp/testMTB/',prefix='SKF_',header=True)
  
##############################################################################
# Predict map
# ---------------------------
raster,_ = datasets.historicalMap()
LAP.predictRaster(raster,'/tmp/classification.tif')

##########################
# Plot example

from matplotlib import pyplot as plt
import gdal
src=gdal.Open('/tmp/classification.tif')
plt.imshow(src.GetRasterBand(1).ReadAsArray(),cmap=plt.get_cmap('tab20'))
plt.axis('off')
plt.show()
