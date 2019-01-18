# -*- coding: utf-8 -*-
"""
Plot confusion matrix
========================================================

Plot confusion matrix from Cross-Validation, with F1 as subplot.

"""

##############################################################################
# Import librairies
# -------------------------------------------
from museotoolbox.learn_tools import learnAndPredict
from museotoolbox.cross_validation import RandomCV
from museotoolbox.charts import plotConfusionMatrix
from museotoolbox import datasets
from sklearn.ensemble import RandomForestClassifier

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
# Initialize Random-Forest
# ---------------------------

classifier = RandomForestClassifier()

##############################################################################
# Start learning
# ---------------------------

LAP = learnAndPredict()
LAP.learnFromRaster(raster,vector,field,cv=RS50,
                    classifier=classifier,param_grid=dict(n_estimators=[100,200]))

##############################################################################
# Get kappa from each fold
# ---------------------------
  
for stats in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
    print(stats['kappa'])

##############################################################################
# Get each confusion matrix from folds
# -----------------------------------------------
cms = []
for stats in LAP.getStatsFromCV(confusionMatrix=True):
    cms.append(stats['confusionMatrix'])
    print(stats['confusionMatrix'])
    
##############################################################################
# Plot confusion matrix
# -----------------------------------------------
    
import numpy as np
meanCM = np.mean(cms,axis=0).astype(np.int16)
pltCM = plotConfusionMatrix(meanCM.T) # Translate for Y = prediction and X = truth
pltCM.addText()
pltCM.colorDiag()

##############################################################################
# Plot confusion matrix and normalize per class
# -----------------------------------------------
from matplotlib.pyplot import cm as colorMap
meanCMnorm = meanCM.astype('float') / meanCM.sum(axis=1)[:, np.newaxis]*100
pltCM = plotConfusionMatrix(meanCMnorm.astype(int).T)
pltCM.addText(alpha_zero=0.8) # in order to hide a little zero values
pltCM.addXlabels(['One','Two','3','Four','Five!'],rotation=90,position='bottom')
pltCM.addYlabels(['','','','','']) # to remove labels
pltCM.colorDiag(diagColor=colorMap.Blues,matrixColor=colorMap.YlOrBr)
