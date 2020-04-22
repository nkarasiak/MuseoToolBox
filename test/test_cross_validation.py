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
from museotoolbox import processing

raster,vector = load_historical_data()
X,y,g = load_historical_data(return_X_y_g=True)
distance_matrix = processing.get_distance_matrix(raster,vector)
n_class = len(np.unique(y,return_counts=True)[1])
smallest_class = np.min(np.unique(y,return_counts=True)[1])



class TestCV(unittest.TestCase):
    def test_train_split(self):
        np.random.seed(42)
        y = np.random.randint(1,3,10).reshape(-1,1)
        X = np.random.randint(1,255,[10,3],dtype=np.uint8)
        g = np.random.randint(1,3,10).reshape(-1,1)

        cv = cross_validation.LeaveOneOut(random_state=42)

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(cv,X,y)
        assert ( X_train.shape[0]+X_test.shape[0] == X.shape[0] )
        assert ( y_train.shape[0]+y_test.shape[0] == y.shape[0] )
        assert ( np.all( np.equal(y_test,np.array([1,2]) ) ) )
        
        X_train, X_test, y_train, y_test, g_train, g_test = cross_validation.train_test_split(cv,X,y,groups=g)
        assert (X_train.shape[0] == y_train.shape[0] == g_train.shape[0])
        assert (X_test.shape[0] == y_test.shape[0] == g_test.shape[0])
        
    def test_loo(self):
        for split in [False,1,2,5]:
            
                cv = cross_validation.LeaveOneOut(n_repeats=split,random_state=split,verbose=split)
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
        cv = cross_validation.RandomStratifiedKFold(valid_size=1/50)        

        self.assertRaises(ValueError,cv.get_n_splits,X,y)
        
        for split in [1,2,5]:
            cv = cross_validation.RandomStratifiedKFold(n_splits=1+split,n_repeats=split,verbose=split)
            assert(cv.get_n_splits(X,y)==split*split+split)
            assert(cv.verbose == split)
            
            for idx,[tr,vl] in enumerate(cv.split(X,y)):
                assert(int(tr.size/vl.size) == split)
                assert(np.unique(y[vl],return_counts=True)[0].size == 5)
        
            assert(idx+1 == split*split+split)
            
    def test_LeavePSubGroupOut(self):
        
        cv = cross_validation.LeavePSubGroupOut(verbose=2)
        for tr,vl in cv.split(X,y,g):
            assert(not np.unique(np.in1d([1,2],[3,4]))[0])

        self.assertRaises(ValueError,cross_validation.LeavePSubGroupOut,valid_size='ko')
        self.assertRaises(ValueError,cross_validation.LeavePSubGroupOut,valid_size=5.1)
        
    def test_LeaveOneSubGroupOut(self):
        cv = cross_validation.LeaveOneSubGroupOut(verbose=2)
        # if only one subgroup
        tempG = np.copy(g)
        tempG[np.where(y==5)] = 1
        self.assertRaises(Exception,cv.get_n_splits,X,y,tempG)
            
        # if all is ok
        cv = cross_validation.LeaveOneSubGroupOut(verbose=2)
        y_vl = np.array([])
        for tr,vl in cv.split(X,y,g):
            y_vl = np.concatenate((y_vl,vl))
            assert(not np.unique(np.in1d([1,2],[3,4]))[0])
        assert(np.all(np.unique(np.asarray(y_vl),return_counts=True)[1]==1))

        list_files =cv.save_to_vector(vector,'Class',group='uniquefid',out_vector='/tmp/cv_g.gpkg')

        assert(len(list_files)==cv.get_n_splits(X,y,g))
        
    def test_SLOO(self):
        
        assert(distance_matrix.shape[0] == y.size)
        
        cv = cross_validation.SpatialLeaveOneOut(distance_thresold=100,
                                                 distance_matrix=distance_matrix,
                                                 random_state=12,verbose=1)
        
        
        processing.sample_extraction(raster,vector,out_vector='/tmp/pixels.gpkg',verbose=False)
        y_ = processing.read_vector_values('/tmp/pixels.gpkg','Class')
        y_polygons = processing.read_vector_values(vector,'Class')
        assert(y_.size == y.size)
        assert(y_polygons.size != y_.size)
        
        list_files=cv.save_to_vector('/tmp/pixels.gpkg','Class',out_vector='/tmp/cv.gpkg')
        assert(len(list_files[0]) == 2)
        for l in list_files:
            for f in l:
                os.remove(f)
        os.remove('/tmp/pixels.gpkg')
        # to keep same size of training by a random selection

            
        as_loo = cross_validation._sample_selection._cv_manager(cross_validation._sample_selection.distanceCV,
                                                            distance_thresold=100,
                                                            distance_matrix=distance_matrix,
                                                            random_state=12,
                                                            LOO_same_size=True,
                                                            valid_size=1)
        y_vl = []
        y_asloo_vl = []
        for sloo_cv,as_loo_cv in zip(cv.split(X,y),as_loo.split(X,y)):
            y_vl.append(sloo_cv[1])
            y_asloo_vl.append(as_loo_cv[1])
            assert(n_class == len(y[sloo_cv[1]]))
            assert(sloo_cv[0].size == as_loo_cv[0].size) # same size between loo and sloo 
            assert(np.all(sloo_cv[1] == as_loo_cv[1])) # using same valid pixel
        
        assert(np.all(np.unique(np.asarray(y_vl),return_counts=True)[1]==1))
        assert(np.all(np.unique(np.asarray(y_asloo_vl),return_counts=True)[1]==1))
        
        as_loo = cross_validation._sample_selection._cv_manager(cross_validation._sample_selection.distanceCV,
                                                            distance_thresold=300,
                                                            distance_matrix=distance_matrix,
                                                            random_state=12,
                                                            LOO_same_size=True,valid_size=2,n_repeats=1,n_splits=5,verbose=1)
        for tr,vl in as_loo.split(X,y):
            assert(vl.size == n_class)
            
        
        as_loo = cross_validation._sample_selection._cv_manager(cross_validation._sample_selection.distanceCV,
                                                            distance_thresold=100,
                                                            distance_matrix=distance_matrix,
                                                            random_state=12,
                                                            LOO_same_size=True,valid_size=False,n_repeats=1,n_splits=5,verbose=1)
        as_loo.get_n_splits(X,y)
        # distance too high 
        cv = cross_validation.SpatialLeaveOneOut(distance_thresold=10000,distance_matrix=distance_matrix,verbose=2)

        self.assertRaises(ValueError,cv.get_n_splits,X,y)            
        
        
    def test_aside(self):
        
        SLOPO = cross_validation.SpatialLeaveAsideOut(valid_size=1/3,
                                     distance_matrix=distance_matrix,random_state=2,verbose=2)
        
        assert(SLOPO.get_n_splits(X,y) == int(1/(1/3)))
            
        for tr,vl in SLOPO.split(X,y):
            assert(np.unique(y[vl]).size == 5) 
            assert(np.unique(y[tr]).size == 5) 
        
        
    def test_slosgo(self)       :
        
        cv = cross_validation.SpatialLeaveOneSubGroupOut(distance_thresold=100,distance_matrix=distance_matrix,distance_label=g,verbose=2)
        
        y_vl = np.array([])
        for tr,vl in cv.split(X,y,g)        :
            print(np.unique(g[vl]))
            assert(n_class==np.unique(g[vl]).size)
        assert(np.all(np.unique(np.asarray(y_vl),return_counts=True)[1]==1))
        
        processing.sample_extraction(raster,vector,out_vector='/tmp/pixels.gpkg',verbose=False)
        test_extensions = ['wrong','shp','gpkg']
        for extension in test_extensions:
            if extension == 'wrong':

                self.assertRaises(Exception,cv.save_to_vector,'/tmp/pixels.gpkg','Class',out_vector='/tmp/SLOSGO.'+extension)
            else:
                list_files = cv.save_to_vector('/tmp/pixels.gpkg','Class',out_vector='/tmp/SLOSGO.'+extension)
                # test overwriting of previous files
                list_files = cv.save_to_vector('/tmp/pixels.gpkg','Class',out_vector='/tmp/SLOSGO.'+extension) 
                for tr,vl in list_files:
                    assert(len(list_files[0]) == 2)
                    for l in list_files:
                        for f in l:     
                            print(f)
                            if os.path.exists(f):
                                os.remove(f)
                            

        
    def test_compare_loo_kf(self):
        cv_loo = cross_validation.LeaveOneOut(random_state=12,verbose=2)
        cv_kf_as_loo = cross_validation.RandomStratifiedKFold(n_splits=False,valid_size=1,random_state=12,verbose=2)
        for trvl_loo,trvl_kf in zip(cv_loo.split(X,y),cv_kf_as_loo.split(X,y)):
            assert(np.all(trvl_loo[0]==trvl_kf[0]))
            assert(np.all(trvl_loo[1]==trvl_kf[1]))
            assert(len(trvl_kf[1]) == n_class)
            assert(np.unique(y[trvl_kf[1]]).size == n_class)
        
        #to print extensions
        cv_loo.get_supported_extensions()
    


if __name__ == "__main__":
    unittest.main()