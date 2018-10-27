#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# ___  ___                       _____           _______           
# |  \/  |                      |_   _|         | | ___ \          
# | .  . |_   _ ___  ___  ___     | | ___   ___ | | |_/ / _____  __
# | |\/| | | | / __|/ _ \/ _ \    | |/ _ \ / _ \| | ___ \/ _ \ \/ /
# | |  | | |_| \__ \  __/ (_) |   | | (_) | (_) | | |_/ / (_) >  < 
# \_|  |_/\__,_|___/\___|\___/    \_/\___/ \___/|_\____/ \___/_/\_\                                                                                                        
#                                             
# @author:  Nicolas Karasiak
# @site:    www.karasiak.net
# @git:     www.github.com/lennepkade/MuseoToolBox
# =============================================================================

from MuseoToolBox.tools import vectorTools,rasterTools
import sklearn

def trainVectorClassifier(X,Y,cv):
    """
    
    """
    pass
    
def tranRasterClassifier(raster,vector,level,cv):
    """
    """
    pass

def modelBasic(classifier='GMM',cv=None,*param_grid):
    """
    
    Parameters
    ----------
    classifier : str or class from scikit-learn.
        str is only 'GMM'. Else, you can input RandomForestClassifier got from 'from sklearn.ensemble import RandomForestClassifier'
    cv : object from MuseoToolBox.crossValidationSelection.samplingSelection()
    """
    if classifier == 'GMM':
        from MuseoToolBox.tools import gmm_ridge as gmmr
        
        try:
            model = gmmr.GMMR()
            model.learn(x,y)
        except:
            raise ChildProcessError('GMM can\'t learn model')
        
    else:
            
        
        from sklearn.model_selection import GridSearchCV
        if cv is None:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=n_splits)
        grid = GridSearchCV(classifier,param_grid=param_grid, cv=cv,n_jobs=n_jobs)
        grid.fit(x,y)
        model = grid.best_estimator_
        model.fit(x,y)
            