# -*- coding: utf-8 -*-
import unittest

import numpy as np
from museotoolbox import ai
from museotoolbox.datasets import load_historical_data
import gdal

import os

from sklearn.ensemble import RandomForestClassifier

raster,vector = load_historical_data(low_res=True)
X,y,g = load_historical_data(return_X_y_g=True,low_res=True)
param_grid = dict(n_estimators=[1,10])
classifier = RandomForestClassifier()


class TestStats(unittest.TestCase):
    def test_SuperLearner(self):
        model = ai.SuperLearner(classifier,param_grid=param_grid,verbose=1,n_jobs=2)
        
        cv = 2
        for tf in [False]:
            model.learn(X,y,group=g,standardize=tf,cv=cv)
            assert(model.predict_array(X).shape == y.shape)
            len(model.CV) == cv
            assert(np.all(model.group == g))
            
            model.predict_image(raster,'/tmp/class.tif',confidence_per_class='/tmp/confclass.tif',higher_confidence='/tmp/higherconf.tif')
            assert(model._array_is_customized == False)
        model.customize_array(np.mean,axis=1)
        model.learn(X,y,group=g,standardize=tf,cv=cv)
        assert(model._array_is_customized == True)
        model.predict_image(raster,'/tmp/class.tif',confidence_per_class='/tmp/confclass.tif',higher_confidence='/tmp/higherconf.tif')
        assert(gdal.Open('/tmp/class.tif').RasterCount == 1)
        assert(gdal.Open('/tmp/higherconf.tif').RasterCount == 1)
        assert(gdal.Open('/tmp/confclass.tif').RasterCount == len(np.unique(y)))
           
    def test_Sequential(self):
        sfs = ai.SequentialFeatureSelection(classifier,param_grid,cv=2)
        sfs.fit(X,y,group=g)
        sfs.predict(X,idx=0)
        assert(not np.all(sfs.predict(X,idx=0) == sfs.predict(X,idx=1)))
        sfs.predict_best_combination(raster,'/tmp/class.tif')
        sfs.predict_images(raster,'/tmp/class')
        
        sfs = ai.SequentialFeatureSelection(classifier,param_grid,cv=2)
        sfs.customize_array(np.mean,axis=-1)
        sfs.fit(X,y,group=g)
        
if __name__ == "__main__":
    unittest.main()
    