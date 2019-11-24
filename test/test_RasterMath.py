# -*- coding: utf-8 -*-
import unittest

import numpy as np
from museotoolbox.raster_tools import RasterMath,rasterize
from museotoolbox.datasets import load_historical_data
from museotoolbox.raster_tools import image_mask_from_vector
from museotoolbox.raster_tools import extract_values
from museotoolbox.vector_tools import read_values
from museotoolbox import learn_tools
import gdal
raster,vector = load_historical_data()

rM = RasterMath(raster)

from sklearn.tree import DecisionTreeClassifier
class TestStringMethods(unittest.TestCase):
    def test_rasterize(self):
        for invert in [True,False]:
            mem = rasterize(raster,vector,'class',out_image='MEM',invert=invert)
            assert(mem.RasterCount == 1)
            assert(mem.RasterXSize == rM.nc)
            assert(mem.RasterYSize == rM.nl)
            
    def test_noImg(self)    :    
        self.assertRaises(ReferenceError,RasterMath,'None',verbose=0)

    def test_dimension(self)    :    
        assert(rM.nb == gdal.Open(raster).RasterCount)
        assert(rM.nl == gdal.Open(raster).RasterYSize)
        assert(rM.nc == gdal.Open(raster).RasterXSize)
        
    
    def test_readPerBand(self):
        for is_3d in [True, False]:
            rM_band = RasterMath(raster,return_3d=is_3d)
            for idx,band in enumerate(rM_band.read_band_per_band()):
                pass
            self.assertEquals(idx+1,rM_band.nb)
            del rM_band

    def test_3d(self)            :
        rM_3d = RasterMath(raster,return_3d=True)
        assert(rM_3d.get_random_block().ndim == 3)
        for block in rM.read_block_per_block():
            pass
        for band in rM.read_band_per_band():
            pass
        rM.custom_block_size(128,256)
        assert(rM.y_block_size==256)
        assert(rM.x_block_size==128)
        rM.add_image(raster)
        
        assert(np.all(rM.get_random_block(12))== np.all(rM.get_random_block(12)))
        
        
    def test_mask(self)            :
        for is_3d in [True, False]:
            mask = '/tmp/mask.tif'
            image_mask_from_vector(vector,raster,mask)
            mask_src = gdal.Open(mask)
            raster_src = gdal.Open(raster)
            assert(mask_src.GetProjection() == raster_src.GetProjection())
            assert(mask_src.RasterCount == 1)
            assert(mask_src.RasterXSize == raster_src.RasterXSize)
            assert(mask_src.RasterYSize == raster_src.RasterYSize)
            rM_band = RasterMath(raster,return_3d=is_3d)
            for idx,band in enumerate(rM_band.read_band_per_band()):
                pass
            assert(idx+1 == rM_band.nb)                        
            x = rM_band.get_random_block()
            assert(x.ndim == is_3d+2)
            

    def test_XYextraction(self):
        X = extract_values(raster,vector)
        assert(X.ndim == 2)
        
        X,y = extract_values(raster,vector,'Class')
        assert(X.shape[0] == y.shape[0])
        
        X,y,g = extract_values(raster,vector,'Class','uniquefid')
        assert(X.shape[0] == y.shape[0] == g.shape[0])
        
    def raster_math_mean(self):
        for is_3d in [True,False]:
            rM = RasterMath(raster,return_3d = is_3d,verbose=is_3d)
            rM.add_function(np.mean,'/tmp/mean.tif',axis=1)
            rM.run()
            assert(gdal.Open('/tmp/mean.tif').RasterCount == 1)
            assert(gdal.Open('/tmp/mean.tif').RasterXSize == rM.nc)
            assert(gdal.Open('/tmp/mean.tif').RasterYSize == rM.nl)
            
            os.remove('/tmp/mean.tif')
            
    def test_unknow_field(self):
        self.assertRaises(ValueError,extract_values,raster,vector,'unknow')
        self.assertRaises(ValueError,read_values,vector)
        
    def test_learn(self):
        LAP = learn_tools.LearnAndPredict(verbose=0)
        LAP.learnFromRaster(raster,vector,'class',classifier=DecisionTreeClassifier())
        assert(LAP.X.ndim == 2)
        assert(LAP.y.shape[0] == LAP.X.shape[0])
        assert(np.all(LAP.model.classes_ == np.unique(LAP.y)))
        assert(LAP.predictArray(LAP.X).shape == LAP.y.shape)
        LAP.predictRaster(raster,'/tmp/map.tif','/tmp/confClass.tif','/tmp/conf.tif')
        assert(gdal.Open('/tmp/map.tif').RasterCount == 1)
        assert(gdal.Open('/tmp/confClass.tif').RasterCount == 5)
        assert(gdal.Open('/tmp/conf.tif').RasterCount == 1)
        
        LAP = learn_tools.LearnAndPredict()
        LAP.customizeX(np.mean,axis=1)
        assert(LAP._x_is_customized == True)
        LAP.learnFromRaster(raster,vector,'class',classifier=DecisionTreeClassifier())
        
        assert(LAP.X.shape[1] == 1)
        
if __name__ == '__main__':
    unittest.main()