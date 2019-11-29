#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 12:03:31 2019

@author: nicolas
"""
# -*- coding: utf-8 -*-
import unittest
import os
import numpy as np

from museotoolbox.datasets import load_historical_data
from museotoolbox import cross_validation
from museotoolbox import geo_tools

raster,vector = load_historical_data()
X,y,g = load_historical_data(return_X_y_g=True)
distance_matrix = geo_tools.get_distance_matrix(raster,vector)
n_class = len(np.unique(y,return_counts=True)[1])
smallest_class = np.min(np.unique(y,return_counts=True)[1])

class TestCV(unittest.TestCase):
    
    def test_loo(self):
        for split in [False,1,2,5]:
            
                cv = cross_validation.LeaveOneOut(n_splits=split,random_state=split,verbose=split)
                if split == False:
                    assert(cv.get_n_splits(X,y)==np.min(np.unique(y,return_counts=True)[-1]))
                else:
                    assert(cv.get_n_splits(X,y)==split)
                assert(cv.verbose == split)
                
                for tr,vl in cv.split(X,y):
                    assert(tr.size == y.size-5)
                    assert(vl.size == 5)
                    assert(len(vl) == 5)
                
            
    def test_kfold(self):
        
        for split in [2,5]:
            cv = cross_validation.RandomStratifiedKFold(n_splits=split,n_repeats=split,verbose=split)
            assert(cv.get_n_splits(X,y)==split*split)
            assert(cv.verbose == split)
            
            for idx,[tr,vl] in enumerate(cv.split(X,y)):
                assert(int(tr.size/vl.size) == split-1)
                assert(np.unique(y[vl],return_counts=True)[0].size == 5)
        
            assert(idx+1 == split*split)
            
    def test_LeavePSubGroupOut(self):
        cv = cross_validation.LeavePSubGroupOut()
        for tr,vl in cv.split(X,y,g):
            assert(not np.unique(np.in1d([1,2],[3,4]))[0])

        self.assertRaises(ValueError,cross_validation.LeavePSubGroupOut,valid_size='ko')
        self.assertRaises(ValueError,cross_validation.LeavePSubGroupOut,valid_size=5)
        
    def test_distanceSLOO(self):
        
        assert(distance_matrix.shape[0] == y.size)
        
        cv = cross_validation.SpatialLeaveOneOut(distance_thresold=100,
                                                 distance_matrix=distance_matrix,
                                                 random_state=12,verbose=1)
        
        geo_tools.sample_extraction(raster,vector,out_vector='/tmp/pixels.gpkg',verbose=False)
        y_ = geo_tools.read_vector_values('/tmp/pixels.gpkg','Class')
        y_polygons = geo_tools.read_vector_values(vector,'Class')
        assert(y_.size == y.size)
        assert(y_polygons.size != y_.size)
        
        list_files=cv.save_to_vector('/tmp/pixels.gpkg','Class',out_vector='/tmp/cv.gpkg')
        assert(len(list_files[0]) == 2)
        for l in list_files:
            for f in l:
                os.remove(f)
        os.remove('/tmp/pixels.gpkg')
        # to keep same size of training by a random selection

            
        as_loo= cross_validation._sample_selection._cv_manager(cross_validation._sample_selection.distanceCV,
                                                            distance_thresold=100,
                                                            distance_matrix=distance_matrix,
                                                            random_state=12,
                                                            LOO_same_size=True,
                                                            valid_size=1)
        
        for sloo_cv,as_loo_cv in zip(cv.split(X,y),as_loo.split(X,y)):
            assert(sloo_cv[0].size == as_loo_cv[0].size) # same size between loo and sloo 
            assert(np.all(sloo_cv[1] == as_loo_cv[1])) # using same valid pixel
            
        
        as_loo= cross_validation._sample_selection._cv_manager(cross_validation._sample_selection.distanceCV,
                                                            distance_thresold=100,
                                                            distance_matrix=distance_matrix,
                                                            random_state=12,
                                                            LOO_same_size=True,valid_size=False)
        for tr,vl in as_loo.split(X,y):
            assert(vl.size == n_class)
            
        # distance too high 
        cv = cross_validation.SpatialLeaveOneOut(distance_thresold=10000,distance_matrix=distance_matrix,verbose=0)
        self.assertRaises(ValueError,cv.get_n_splits,X,y)
            
        
        
    def test_aside(self):
        
        SLOPO = cross_validation.SpatialLeaveAsideOut(valid_size=1/3,
                                     distance_matrix=distance_matrix,random_state=2)
        
        assert(SLOPO.get_n_splits(X,y) == int(1/(1/3)))
        
        for tr,vl in SLOPO.split(X,y):
            assert(np.unique(y[vl]).size == 5) 
            assert(np.unique(y[tr]).size == 5) 
        
        
    def test_SLOSGO(self)       :
        cv = cross_validation.SpatialLeaveOneSubGroupOut(distance_thresold=100,distance_matrix=distance_matrix,distance_label=g)
        for tr,vl in cv.split(X,y,g)        :
            assert(n_class==np.unique(g[vl]).size)
        
        geo_tools.sample_extraction(raster,vector,out_vector='/tmp/pixels.gpkg',verbose=False)
        test_extensions = ['wrong','shp','gpkg']
        for extension in test_extensions:
            if extension == 'wrong':
                self.assertRaises(Exception,cv.save_to_vector,'/tmp/pixels.gpkg','Class',out_vector='/tmp/SLOSGO.'+extension)
            else:
                list_files = cv.save_to_vector('/tmp/pixels.gpkg','Class',out_vector='/tmp/SLOSGO.'+extension)
                for tr,vl in list_files:
                    assert(len(list_files[0]) == 2)
                    for l in list_files:
                        for f in l:                    
                            if os.path.exists(f):
                                os.remove(f)
                            
        os.remove('/tmp/pixels.gpkg')

        
    def test_LOO(self):
        cv_loo = cross_validation.LeaveOneOut(random_state=12)
        cv_kf_as_loo = cross_validation.RandomStratifiedKFold(n_splits=False,valid_size=1,random_state=12)
        for trvl_loo,trvl_kf in zip(cv_loo.split(X,y),cv_kf_as_loo.split(X,y)):
            assert(np.all(trvl_loo[0]==trvl_kf[0]))
            assert(np.all(trvl_loo[1]==trvl_kf[1]))
            assert(len(trvl_kf[1]) == n_class)
            assert(np.unique(y[trvl_kf[1]]).size == n_class)
        
        #to print extensions
        cv_loo.get_supported_extensions()
    


if __name__ == "__main__":
    unittest.main()