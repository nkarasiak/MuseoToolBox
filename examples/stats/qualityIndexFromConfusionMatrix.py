# -*- coding: utf-8 -*-
"""
Compute quality index from confusion matrix
===============================================================

Compute different quality index  (OA, Kappa and F1) directly
from confusion matrix.

"""

##############################################################################
# Import librairies
# -------------------------------------------
import numpy as np
from museotoolbox.stats import ConfusionMatrixStats
from museotoolbox.charts import PlotConfusionMatrix

##############################################################################
# Create a random confusion matrix
# -------------------------------------------

confusion_matrix = np.random.randint(1,50,size=[6,6])
print('Total number of pixels : '+str(np.sum(confusion_matrix)))

PlotConfusionMatrix(confusion_matrix).add_text()

##############################################################################
# Generate index from the confusion matrix

sts = ConfusionMatrixStats(confusion_matrix=confusion_matrix)

################################
# show quality
print('OA  : ' +str(sts.OA))
print('F1 per class : '+str(sts.F1))
print('F1mean : '+str(sts.F1mean))
print('Kappa : '+str(sts.kappa))