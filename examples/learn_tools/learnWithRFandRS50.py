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

from museotoolbox.learn_tools import learnAndPredict
from museotoolbox.cross_validation import RandomCV
from museotoolbox import datasets
from museotoolbox import raster_tools
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.historicalMap()
field = 'Class'

##############################################################################
# Create CV
# -------------------------------------------

RS50 = RandomCV(valid_size=0.5,n_splits=2,
                random_state=12,verbose=False)

##############################################################################
# Initialize Random-Forest and metrics
# --------------------------------------

classifier = RandomForestClassifier(random_state=12,n_jobs=-1)

# 
kappa = metrics.make_scorer(metrics.cohen_kappa_score)
f1_mean = metrics.make_scorer(metrics.f1_score,average='micro')
scoring = dict(kappa=kappa,f1_mean=f1_mean,accuracy='accuracy')

##############################################################################
# Start learning
# ---------------------------
# sklearn will compute different metrics, but will keep best results from kappa (refit='kappa')
LAP = learnAndPredict(n_jobs=-1,verbose=1)
LAP.learnFromRaster(raster,vector,field,cv=RS50,
                    classifier=classifier,param_grid=dict(n_estimators=[100,200]),
                    scoring=scoring,refit='kappa')

##############################################################################
# Read the model
# -------------------
print(LAP.model)
print(LAP.model.cv_results_)
print(LAP.model.best_score_)


##############################################################################
# Get F1 for every class from best params
# -----------------------------------------------

for stats in LAP.getStatsFromCV(confusionMatrix=False,F1=True):
    print(stats['F1'])
    
##############################################################################
# Get each confusion matrix from folds
# -----------------------------------------------

for stats in LAP.getStatsFromCV(confusionMatrix=True):
    print(stats['confusionMatrix'])
    
##############################################################################
# Save each confusion matrix from folds
# -----------------------------------------------

LAP.saveCMFromCV('/tmp/testMTB/',prefix='RS50_')

##############################################################################
# Predict map
# ---------------------------
    
LAP.predictRaster(raster,'/tmp/classification.tif',
                  confidence='/tmp/confidence.tif',
                  confidencePerClass='/tmp/confidencePerClass.tif')

##########################
# Plot example

from matplotlib import pyplot as plt
import gdal
src=gdal.Open('/tmp/classification.tif')
plt.imshow(src.GetRasterBand(1).ReadAsArray(),cmap=plt.get_cmap('tab20'))
plt.axis('off')
plt.show()
