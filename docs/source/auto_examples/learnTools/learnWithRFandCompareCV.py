# -*- coding: utf-8 -*-
"""
Learn with Random-Forest and  compare Cross-Validation methods
===============================================================

This example shows how to make a classification with different cross-validation methods

"""

##############################################################################
# Import librairies
# -------------------------------------------

from MuseoToolBox.crossValidationTools import RandomCV,LeavePSubGroupOut,LeaveOneSubGroupOut
from MuseoToolBox import datasets,rasterTools,vectorTools
from MuseoToolBox import learnTools
from sklearn.ensemble import RandomForestClassifier

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.getHistoricalMap()
field = 'Class'
group = 'uniquefid'

##############################################################################
# Initialize Random-Forest
# ---------------------------

classifier = RandomForestClassifier()

##############################################################################
# Initialize Random-Forest
# ---------------------------

CVs = [RandomCV(),LeavePSubGroupOut(),LeaveOneSubGroupOut()]
kappas=[]
for cv in CVs : 
    LAP = learnTools.learnAndPredict()
    
    ##############################################################################
    # Define group only when needed
    
    if cv == RandomCV():
        group=None
    else:
        group=group
        
    LAP.learnFromRaster(raster,vector,inField=field,inGroup=group,cv=cv,
                        classifier=classifier,param_grid=dict(n_estimators=[100,200]))
    print('Kappa for '+str(type(RandomCV()).__name__))
    cvKappa = []
    
    for kappa in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
        print(kappa)
        cvKappa.append(kappa)
    
    kappas.append(cvKappa)
    
    print(20*'=')

##########################
# Plot example


from matplotlib import pyplot as plt
plt.title('Kappa according to Cross-validation methods')
plt.boxplot(kappas,labels=[str(type(i).__name__) for i in CVs], patch_artist=True)
plt.grid()
plt.ylabel('Kappa')
plt.show()
