# -*- coding: utf-8 -*-
import unittest

import numpy as np
from museotoolbox import ai
from museotoolbox.datasets import load_historical_data
import gdal

import os
import shutil

from sklearn.ensemble import RandomForestClassifier

raster,vector = load_historical_data(low_res=True)
X,y,g = load_historical_data(return_X_y_g=True,low_res=True)
param_grid = dict(n_estimators=[1,10])
classifier = RandomForestClassifier()


class TestStats(unittest.TestCase):
    def test_superLearner(self):
        
        n_cv = 2
        for tf in [True,False]:
            model = ai.SuperLearner(classifier,param_grid=param_grid,verbose=tf,n_jobs=2)
            model.fit(X,y,group=g,standardize=tf,cv=n_cv)
            assert(model.predict_array(X).shape == y.shape)
            len(model.CV) == n_cv
            assert(np.all(model.group == g))
            
            model.predict_image(raster,'/tmp/SuperLearner/class.tif',confidence_per_class='/tmp/SuperLearner/confclass.tif',higher_confidence='/tmp/SuperLearner/higherconf.tif')
            assert(model._is_standardized == tf)
        
        model.customize_array(np.mean,axis=1)
        assert(model._array_is_customized)
        assert(model.xFunction)
        assert(np.all(model.standardize_array(X) != X))
        model.standardize_array()
        model.fit(X,y,group=g,standardize=tf,cv=n_cv)
        assert(model._array_is_customized == True)
        model.predict_image(raster,'/tmp/SuperLearner/class.tif',confidence_per_class='/tmp/SuperLearner/confclass.tif',higher_confidence='/tmp/SuperLearner/higherconf.tif')
        assert(gdal.Open('/tmp/SuperLearner/class.tif').RasterCount == 1)
        assert(gdal.Open('/tmp/SuperLearner/higherconf.tif').RasterCount == 1)
        assert(gdal.Open('/tmp/SuperLearner/confclass.tif').RasterCount == len(np.unique(y)))
        cms = model.get_stats_from_cv()
        
        assert(len(cms) == n_cv)
        
        model.save_model('/tmp/SuperLearner/model.npz')
        assert(os.path.exists('/tmp/SuperLearner/model.npz'))
        model.load_model('/tmp/SuperLearner/model.npz')
        shutil.rmtree('/tmp/SuperLearner/')

        
    def test_sequential(self):
        sfs = ai.SequentialFeatureSelection(classifier,param_grid,cv=2)
        sfs.fit(X,y)
        sfs.predict(X,idx=0)
        assert(not np.all(sfs.predict(X,idx=0) == sfs.predict(X,idx=1)))
        sfs.predict_best_combination(raster,'/tmp/class.tif')
        sfs.predict_images(raster,'/tmp/class')
        
        n_comp = 2
        max_features = 2
        sfs = ai.SequentialFeatureSelection(classifier,param_grid,cv=2,n_comp=n_comp,path_to_save_models='/tmp/sfs_models/')
        def double_columns(x):
            return np.hstack((x,x))
        sfs.customize_array(double_columns)
        sfs.fit(X,y,max_features=max_features,standardize=True)
        assert(sfs.X.shape[1] == X.shape[1]*2)
        assert(len(sfs.best_features_) == 2)
        assert(sfs.get_best_model().X.shape[1] == n_comp*max_features        )
        shutil.rmtree('/tmp/sfs_models/')
        
if __name__ == "__main__":
    unittest.main()
    