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
from museotoolbox import vector_tools
from museotoolbox import raster_tools

import gdal
raster,vector = load_historical_data()
X,y = load_historical_data(return_X_y=True)

from sklearn.tree import DecisionTreeClassifier
class TestStringMethods(unittest.TestCase):

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
            
    def test_distanceSLOO(self):
        X,y = raster_tools.extract_values(raster,vector,'Class')
        distance_matrix = vector_tools.get_distance_matrix(raster,vector)
        assert(distance_matrix.shape[0] == y.size)
        for split in [2]:
#            cv = cross_validation.SpatialLeaveOneOut(distanceThresold=100,distanceMatrix=distance_matrix,verbose=split,random_state=12)
            cv = cross_validation.SpatialLeaveOneOut(distanceThresold=100,distanceMatrix=distance_matrix,random_state=12)
            
            print(cv.get_n_splits(X,y))
            
            
            ###############################################################################
            # .. note::
            #    Split is made to generate each fold
            
#            for tr,vl in cv.split(X,y):
#                print(tr.shape,vl.shape)

            vector_tools.sample_extraction(raster,vector,out_vector='/tmp/pixels.gpkg',verbose=False)
            
            list_files=cv.save_to_vector('/tmp/pixels.gpkg','Class',out_vector='/tmp/cv.gpkg')
            assert(len(list_files[0]) == 2)
            for l in list_files:
                for f in l:
                    os.remove(f)
            os.remove('/tmp/pixels.gpkg')
            
                    
    def test_aside(self):
        
        
        distance_matrix = vector_tools.get_distance_matrix(raster,vector)
        X,y = raster_tools.extract_values(raster,vector,'Class')
        ##############################################################################
        # Create CV
        # -------------------------------------------
        # n_splits will be the number  of the least populated class
        
        SLOPO = cross_validation.SpatialLeaveAsideOut(valid_size=1/3,n_splits=2,
                                     distanceMatrix=distance_matrix,random_state=2)
        
        print(SLOPO.get_n_splits(X,y))
        
        ###############################################################################
        # .. note::
        #    Split is made to generate each fold
        
        for tr,vl in SLOPO.split(X,y):
            assert(np.unique(y[vl]).size == 5) 
            assert(np.unique(y[tr]).size == 5) 
            
        
        ###############################################################################
        #    Save each train/valid fold in a file
        # -------------------------------------------
        # In order to translate polygons into points (each points is a pixel in the raster)
        # we use sampleExtraction from vector_tools to generate a temporary vector.
        
        vector_tools.sample_extraction(raster,vector,out_vector='/tmp/pixels.gpkg',verbose=False)
        list_files = SLOPO.save_to_vector('/tmp/pixels.gpkg','Class',out_vector='/tmp/SLOPO.gpkg')
        for tr,vl in list_files:
            assert(len(list_files[0]) == 2)
            for l in list_files:
                for f in l:
                    if os.path.exists(f):
                        os.remove(f)
                    
        os.remove('/tmp/pixels.gpkg')
            
          
if __name__ == '__main__':
    unittest.main()