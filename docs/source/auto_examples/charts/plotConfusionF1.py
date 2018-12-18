# -*- coding: utf-8 -*-
"""
Plot confusion matrix from Cross-Validation with F1
========================================================

Plot confusion matrix from Cross-Validation, with F1 as subplot.

"""

##############################################################################
# Import librairies
# -------------------------------------------
from museotoolbox.learnTools import learnAndPredict
from museotoolbox.crossValidation import RandomCV
from museotoolbox.charts import plotConfusionMatrix
from museotoolbox import datasets
from sklearn.ensemble import RandomForestClassifier

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.getHistoricalMap()
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
                    classifier=classifier,param_grid=dict(n_estimators=[10,100]))

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
pltCM.addF1()
pltCM.colorDiag()
pltCM.show()

##############################################################################
# Plot confusion matrix and normalize per class
# -----------------------------------------------

meanCM = meanCM.astype('float') / meanCM.sum(axis=1)[:, np.newaxis]*100
pltCM = plotConfusionMatrix(meanCM.astype(int).T)
pltCM.addText(alpha_zero=0.3) # in order to hide a little zero values
pltCM.addF1()
pltCM.colorDiag()
pltCM.show()
