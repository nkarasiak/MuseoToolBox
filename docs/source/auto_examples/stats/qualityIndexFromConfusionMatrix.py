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
from museotoolbox.stats import retrieve_y_from_confusion_matrix
from museotoolbox.charts import PlotConfusionMatrix
from sklearn.metrics import accuracy_score,cohen_kappa_score
##############################################################################
# Create a random confusion matrix
# -------------------------------------------

confusion_matrix = np.random.randint(1,30,size=[6,6])
confusion_matrix[range(6),range(6)] += 40
print('Total number of pixels : '+str(np.sum(confusion_matrix)))

PlotConfusionMatrix(confusion_matrix).add_text()

##############################################################################
# Generate index from the confusion matrix

yp,yt = retrieve_y_from_confusion_matrix(confusion_matrix)

################################
# show quality
print('OA is : '+str(accuracy_score(yp,yt)))
print('Kappa is : '+str(cohen_kappa_score(yp,yt)))