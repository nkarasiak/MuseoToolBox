# -*- coding: utf-8 -*-
"""
Plot confusion matrix with User/Producer accuracy
========================================================

Plot confusion matrix from Cross-Validation, with accuracy (user/prod) as subplot.

"""

##############################################################################
# Import librairies
# -------------------------------------------
from museotoolbox.learn_tools import learnAndPredict
from museotoolbox.cross_validation import RandomStratifiedKFold
from museotoolbox.charts import plotConfusionMatrix
from museotoolbox import datasets
from sklearn.ensemble import RandomForestClassifier

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.historicalMap(low_res=True)
field = 'Class'

##############################################################################
# Create CV
# -------------------------------------------
RSKF = RandomStratifiedKFold(n_splits=2,
                random_state=12,verbose=False)

##############################################################################
# Initialize Random-Forest
# ---------------------------

classifier = RandomForestClassifier()

##############################################################################
# Start learning
# ---------------------------


LAP = learnAndPredict()
LAP.learnFromRaster(raster,vector,field,cv=RSKF,
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

# a bug in Sphinx doesn't show the whole plot, sorry.

labels = ['Forest','Agriculture','Bare soil','Water','Building']
from matplotlib.pyplot import cm as colorMap
meanCM = np.mean(cms,axis=0).astype(np.int16)
pltCM = plotConfusionMatrix(meanCM.T) # Translate for Y = prediction and X = truth
pltCM.addText()
pltCM.addXlabels(labels,rotation=90)
pltCM.addYlabels(labels)
pltCM.colorDiag(diagColor=colorMap.Purples,matrixColor=colorMap.Reds)
pltCM.addAccuracy()