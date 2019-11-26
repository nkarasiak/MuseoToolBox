# -*- coding: utf-8 -*-
import unittest

import numpy as np
from museotoolbox import geo_tools
from museotoolbox.datasets import load_historical_data
from museotoolbox import ai
import gdal


import os
from sklearn.tree import DecisionTreeClassifier

raster,vector = load_historical_data()
rM =geo_tools.RasterMath(raster)

class TestRaster(unittest.TestCase):
    def test_np_to_gdt(self):
        
        self._assert_np_gdt(np.dtype('uint8').name,gdal.GDT_Byte)
        self._assert_np_gdt(np.dtype('int16').name,gdal.GDT_Int16)
        self._assert_np_gdt(np.dtype('uint16').name,gdal.GDT_UInt16)
        self._assert_np_gdt(np.dtype('int32').name,gdal.GDT_Int32)
        self._assert_np_gdt(np.dtype('uint32').name,gdal.GDT_UInt32)
        
        self._assert_np_gdt(np.dtype('int64').name,gdal.GDT_Int32)
        self._assert_np_gdt(np.dtype('uint64').name,gdal.GDT_Int32)
        
        self._assert_np_gdt(np.dtype('uint16').name,gdal.GDT_UInt16)
        self._assert_np_gdt(np.dtype('float32').name,gdal.GDT_Float32)
        self._assert_np_gdt(np.dtype('float64').name,gdal.GDT_Float64)
        
        self._assert_np_gdt(gdal.GDT_Byte,np.uint8)
        self._assert_np_gdt(gdal.GDT_Int16,np.int16)
        self._assert_np_gdt(gdal.GDT_UInt16,np.uint16)
        self._assert_np_gdt(gdal.GDT_Float64,np.float64)
        self._assert_np_gdt(gdal.GDT_Float32,np.float32)
        
        self._assert_np_gdt(np.dtype('float128').name,gdal.GDT_Float64)
        
        
    def _assert_np_gdt(self,in_conv,out_dt):
        assert(geo_tools.convert_dt(in_conv)==out_dt)
        
    def test_gdt_minmax_values(self):
        assert(gdal.GDT_UInt16 == geo_tools.get_gdt_from_minmax_values(500))
        assert(gdal.GDT_Int16 == geo_tools.get_gdt_from_minmax_values(max_value=1,min_value=-5))
        assert(gdal.GDT_Float32 == geo_tools.get_gdt_from_minmax_values(max_value=2,min_value=-55.55))
        assert(gdal.GDT_Byte == geo_tools.get_gdt_from_minmax_values(max_value=222))
        assert(gdal.GDT_Float64 == geo_tools.get_gdt_from_minmax_values(max_value =888E+40))
        assert(gdal.GDT_Float64 == geo_tools.get_gdt_from_minmax_values(max_value=5,min_value = -888E+40))
        
    def test_rasterize(self):
        for invert in [True,False]:
            for field in ['class',False]:
                mem = geo_tools.rasterize(raster,vector,field,out_image='MEM',invert=invert)
                assert(mem.RasterCount == 1)
                assert(mem.RasterXSize == rM.n_columns)
                assert(mem.RasterYSize == rM.n_lines)
            
    def test_noImg(self)    :    
        
        self.assertRaises(ReferenceError,geo_tools.RasterMath,'None',verbose=0)
        
    def test_dimension(self)    :    
        assert(rM.n_bands == gdal.Open(raster).RasterCount)
        assert(rM.n_lines == gdal.Open(raster).RasterYSize)
        assert(rM.n_columns == gdal.Open(raster).RasterXSize)
        
    
    def test_readPerBand(self):
        for is_3d in [True, False]:
            rM_band = geo_tools.RasterMath(raster,return_3d=is_3d)
            for idx,band in enumerate(rM_band.read_band_per_band()):
                pass
            assert(idx+1==rM_band.n_bands)
            del rM_band
    
    def test_3d(self)            :
        rM_3d = geo_tools.RasterMath(raster,return_3d=True)
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
            geo_tools.image_mask_from_vector(vector,raster,mask)
            mask_src = gdal.Open(mask)
            raster_src = gdal.Open(raster)
            assert(mask_src.GetProjection() == raster_src.GetProjection())
            assert(mask_src.RasterCount == 1)
            assert(mask_src.RasterXSize == raster_src.RasterXSize)
            assert(mask_src.RasterYSize == raster_src.RasterYSize)
            rM_band = geo_tools.RasterMath(raster,return_3d=is_3d)
            for idx,band in enumerate(rM_band.read_band_per_band()):
                pass
            assert(idx+1 == rM_band.n_bands)                        
            x = rM_band.get_random_block()
            assert(x.ndim == is_3d+2)
            
    
    def test_XYextraction(self):
        X = geo_tools.extract_ROI(raster,vector)
        assert(X.ndim == 2)
        
        X,y = geo_tools.extract_ROI(raster,vector,'Class')
        assert(X.shape[0] == y.shape[0])
        
        X,y,g = geo_tools.extract_ROI(raster,vector,'Class','uniquefid')
        assert(X.shape[0] == y.shape[0] == g.shape[0])
    
        self.assertRaises(ValueError,geo_tools.extract_ROI,'wrong/path','wrong/path/too')
        assert(geo_tools.extract_ROI(raster,vector).shape[1] == gdal.Open(raster).RasterCount)
        self.assertRaises(ValueError,geo_tools.extract_ROI,raster,vector,'kodidk')
        
        
        
    def raster_math_mean(self):
        for is_3d in [True,False]:
            rM = geo_tools.RasterMath(raster,return_3d = is_3d,verbose=is_3d)
            rM.add_function(np.mean,'/tmp/mean.tif',axis=1)
            rM.run()
            assert(gdal.Open('/tmp/mean.tif').RasterCount == 1)
            assert(gdal.Open('/tmp/mean.tif').RasterXSize == rM.n_columns)
            assert(gdal.Open('/tmp/mean.tif').RasterYSize == rM.n_lines)
            
            os.remove('/tmp/mean.tif')
            
    def test_unknow_field(self):
        self.assertRaises(ValueError,geo_tools.extract_ROI,raster,vector,'unknow')
        self.assertRaises(ValueError,geo_tools.read_vector_values,vector)
            
#    def test_moran(self):
#        # Maybe generate a false raster image to validate Moran's I
#        im_mask = image_mask_from_vector(vector,raster,'/tmp/mask.tif')
#        
#        for mask in [im_mask,False]:
#            res = []
#            for tf_lag in [True,False]:
#                for standardisation in ['b','r']:
#                    i_moran=Moran(raster,mask,transform=standardisation,lag=2,intermediate_lag=tf_lag)
#                    res.append(np.mean(i_moran.scores['I']))
#            assert(np.unique(res).size == len(res)*(tf_lag+1)) # every moran I is different (lag,row standardisation, mask)
#                
if __name__ == "__main__":
    unittest.main()
    