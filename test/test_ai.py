# -*- coding: utf-8 -*-
import unittest

import numpy as np
from museotoolbox import ai
from museotoolbox.datasets import load_historical_data
from museotoolbox.processing import image_mask_from_vector
from osgeo import gdal

import os
import shutil

from sklearn.ensemble import RandomForestClassifier

raster,vector = load_historical_data(low_res=True)
X,y,g = load_historical_data(return_X_y_g=True,low_res=True)
param_grid = dict(n_estimators=[1,10])
classifier = RandomForestClassifier()
image_mask_from_vector(vector,raster,'/tmp/mask.tif')

class TestStats(unittest.TestCase):
    def test_superLearner(self):
        
        n_cv = 2
        for tf in [True,False]:
            verbose = tf+1
            model = ai.SuperLearner(classifier,param_grid=param_grid,n_jobs=1,verbose=verbose)
            model.fit(X,y,group=g,standardize=tf,cv=n_cv)
            assert(model.predict_array(X).shape == y.shape)
            len(model.CV) == n_cv
            assert(np.all(model.group == g))
            
            model.predict_image(raster,'/tmp/SuperLearner/class.tif',confidence_per_class='/tmp/SuperLearner/confclass.tif',higher_confidence='/tmp/SuperLearner/higherconf.tif')
            assert(model._is_standardized == tf)
        
        # test masked return if X is totally masked
        X_masked = np.ma.copy(X)
        X_masked.mask=True
        X_masked_return = model._convert_array(X_masked)
        assert(np.ma.is_masked(X_masked_return))
        

        
    def test_superLearn_pred(self):
        model = ai.SuperLearner(classifier,param_grid=param_grid,n_jobs=1,verbose=0)
        model.customize_array(np.mean,axis=1)
        model.fit(X,y,group=g,standardize=True,cv=2)
#        #    
        assert(model._array_is_customized == True)
        assert(model._array_is_customized)
        assert(model.xFunction)
        assert(np.all(model.standardize_array(X) != X))
        model.predict_image(raster,out_image='/tmp/SuperLearner/class.tif',in_image_mask='/tmp/mask.tif',confidence_per_class='/tmp/SuperLearner/confclass.tif',higher_confidence='/tmp/SuperLearner/higherconf.tif')
        assert(gdal.Open('/tmp/SuperLearner/class.tif').RasterCount == 1)
        assert(gdal.Open('/tmp/SuperLearner/higherconf.tif').RasterCount == 1)
        assert(gdal.Open('/tmp/SuperLearner/confclass.tif').RasterCount == len(np.unique(y)))
        cms = model.get_stats_from_cv()
        
        assert(len(cms) == 2)
        model.save_cm_from_cv('/tmp/empty/',prefix='coco',header=False)
        shutil.rmtree('/tmp/empty/')
        model.save_model('/tmp/SuperLearner/model')
        assert(os.path.exists('/tmp/SuperLearner/model.npz'))
        model.load_model('/tmp/SuperLearner/model')
        shutil.rmtree('/tmp/SuperLearner/')

        with self.assertRaises(ValueError):
            model.fit(X,y,cv=False)
        
    def test_sequential(self):
        
        sfs = ai.SequentialFeatureSelection(classifier,param_grid)
        sfs.fit(X,y,cv=2)
        sfs.predict(X,idx=0)
        assert(not np.all(sfs.predict(X,idx=0) == sfs.predict(X,idx=1)))
        sfs.predict_best_combination(raster,'/tmp/class.tif')
        sfs.predict_images(raster,'/tmp/class')
        assert(sfs.get_best_model())
        assert(sfs.transform(X,idx='best').shape[1] == sfs.best_idx_+1)

        n_comp = 2
        max_features = 2
        sfs = ai.SequentialFeatureSelection(classifier,param_grid,n_comp=n_comp,path_to_save_models='/tmp/sfs_models/',verbose=1)
        def double_columns(x):
            return np.hstack((x,x))
        sfs.customize_array(double_columns)
        sfs.fit(X,y,max_features=max_features,standardize=True,cv=2)
        sfs.fit(X,y,max_features=max_features,standardize=True,cv=2) # to reload from path
        assert(sfs.transform(X,idx=1).shape[1] == 2*n_comp)
        assert(sfs.transform(X,idx=0).shape[1] == n_comp)
        assert(sfs.X.shape[1] == X.shape[1]*2)
        assert(len(sfs.best_features_) == 2)
        sfs.predict_images(raster,'/tmp/sfs_models/')
        sfs.predict_best_combination(raster,'/tmp/sfs_models/best.tif')
        assert(sfs.get_best_model().X.shape[1] == n_comp*(sfs.best_idx_+1) )
        sfs.predict(X,0)
        shutil.rmtree('/tmp/sfs_models/')
        
if __name__ == "__main__":
    unittest.main()
    