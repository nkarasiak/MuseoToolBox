# -*- coding: utf-8 -*-
"""
Learn algorithm and customize your input raster without writing it on disk
=============================================================================

This example shows how to customize your raster (ndvi, smooth signal...) in the 
learning process to avoi generate a new raster.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.ai import SuperLearner
from museotoolbox.processing import extract_ROI
from museotoolbox import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.load_historical_data(low_res=True)
field = 'Class'

##############################################################################
# Initialize Random-Forest and metrics
# --------------------------------------

classifier = RandomForestClassifier(random_state=12,n_jobs=1)

kappa = metrics.make_scorer(metrics.cohen_kappa_score)
f1_mean = metrics.make_scorer(metrics.f1_score,average='micro')
scoring = dict(kappa=kappa,f1_mean=f1_mean,accuracy='accuracy')


##############################################################################
# Start learning
# ---------------------------
# sklearn will compute different metrics, but will keep best results from kappa (refit='kappa')
SL = SuperLearner(classifier=classifier,param_grid=dict(n_estimators=[10]),n_jobs=1,verbose=1)

##############################################################################
# Create or use custom function

def reduceBands(X,bandToKeep=[0,2]):
    # this function get the first and the last band
    X=X[:,bandToKeep].reshape(-1,len(bandToKeep))
    return X

# add this function to learnAndPredict class
SL.customize_array(reduceBands)

# if you learn from vector, refit according to the f1_mean
X,y = extract_ROI(raster,vector,field)
SL.fit(X,y,cv=2,scoring=scoring,refit='f1_mean')

##############################################################################
# Read the model
# -------------------
print(SL.model)
print(SL.model.cv_results_)
print(SL.model.best_score_)

##############################################################################
# Get F1 for every class from best params
# -----------------------------------------------

for stats in SL.get_stats_from_cv(confusion_matrix=False,F1=True):
    print(stats['F1'])
    
##############################################################################
# Get each confusion matrix from folds
# -----------------------------------------------

for stats in SL.get_stats_from_cv(confusion_matrix=True):
    print(stats['confusion_matrix'])
    
##############################################################################
# Save each confusion matrix from folds
# -----------------------------------------------

SL.save_cm_from_cv('/tmp/testMTB/',prefix='RS50_')

##############################################################################
# Predict map
# ---------------------------
    
SL.predict_image(raster,'/tmp/classification.tif',
                  higher_confidence='/tmp/confidence.tif',
                  confidence_per_class='/tmp/confidencePerClass.tif')
##########################
# Plot example

from matplotlib import pyplot as plt
from osgeo import gdal
src=gdal.Open('/tmp/classification.tif')
plt.imshow(src.GetRasterBand(1).ReadAsArray(),cmap=plt.get_cmap('tab20'))
plt.axis('off')
plt.show()
