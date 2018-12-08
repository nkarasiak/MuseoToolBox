# -*- coding: utf-8 -*-
"""
Plot confusion matrix from Cross-Validation
========================================================

Plot confusion matrix from Cross-Validation, and subplot User/Producer`s
accuracy or F1.

"""

##############################################################################
# Import librairies
# -------------------------------------------
from MuseoToolBox.learnTools import learnAndPredict
from MuseoToolBox.crossValidationTools import RandomCV
from MuseoToolBox.chartsTools import plotConfusionMatrix
from MuseoToolBox import datasets
from sklearn.ensemble import RandomForestClassifier

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.getHistoricalMap()
field = 'Class'
##############################################################################
# Create CV
# -------------------------------------------
RS50 = RandomCV(valid_size=0.5,n_splits=10,
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
  
for kappa in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
    print(kappa)

##############################################################################
# Get each confusion matrix from folds
# -----------------------------------------------
cms = []
for cm in LAP.getStatsFromCV(confusionMatrix=True):
    cms.append(cm)
    print(cm)
    
##############################################################################
# Plot confusion matrix
# -----------------------------------------------
    
import numpy as np
meanCM = np.mean(cms,axis=0)[0,:,:].astype(np.int16)
pltCM = plotConfusionMatrix(meanCM.T) # Translate for Y = prediction and X = truth
pltCM.addText(thresold=1500)
pltCM.addF1()
pltCM.colorDiag()

##############################################################################
# Plot confusion matrix and normalize per class
# -----------------------------------------------

meanCM = meanCM.astype('float') / meanCM.sum(axis=1)[:, np.newaxis]*100
pltCM = plotConfusionMatrix(meanCM.astype(int).T)
pltCM.addText(thresold=50)
pltCM.addF1()
pltCM.colorDiag()
