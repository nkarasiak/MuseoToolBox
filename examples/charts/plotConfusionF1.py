# -*- coding: utf-8 -*-
"""
Plot confusion matrix from Cross-Validation with F1
========================================================

Plot confusion matrix from Cross-Validation, with F1 as subplot.

"""

##############################################################################
# Import librairies
# -------------------------------------------
from museotoolbox.ai import SuperLearner
from museotoolbox.cross_validation import RandomStratifiedKFold
from museotoolbox.charts import PlotConfusionMatrix
from museotoolbox import datasets
from sklearn.ensemble import RandomForestClassifier

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

X,y = datasets.load_historical_data(low_res=True,return_X_y=True)
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


SL = SuperLearner(classifier=classifier,param_grid=dict(n_estimators=[10,50]))
SL.fit(X,y,cv=RSKF)

##############################################################################
# Get kappa from each fold
# ---------------------------
  
for stats in SL.get_stats_from_cv(confusion_matrix=False,kappa=True):
    print(stats['kappa'])

##############################################################################
# Get each confusion matrix from folds
# -----------------------------------------------
cms = []
for stats in SL.get_stats_from_cv(confusion_matrix=True):
    cms.append(stats['confusion_matrix'])
    print(stats['confusion_matrix'])
    
##############################################################################
# Plot confusion matrix
# -----------------------------------------------
    
import numpy as np
meanCM = np.mean(cms,axis=0).astype(np.int16)
pltCM = PlotConfusionMatrix(meanCM.T) # Translate for Y = prediction and X = truth
pltCM.add_text()
pltCM.add_f1()
pltCM.color_diagonal()