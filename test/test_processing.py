# -*- coding: utf-8 -*-
import unittest
from shutil import copyfile
import numpy as np
from museotoolbox import processing
from museotoolbox.datasets import load_historical_data
from osgeo import gdal, osr
import os

raster,vector = load_historical_data()
rM = processing.RasterMath(raster)
mask = processing.image_mask_from_vector(vector,raster,'/tmp/mask.tif')


def create_false_image(array,path):
    # from https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(path, array.shape[1], array.shape[0], 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform((0, 10, 0, 0, 0, 10))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

# create autocorrelated tif
x = np.zeros((100,100),dtype=int)
# max autocorr
x[:50,:] = 1
x[50:,:] = 2

x_mask = np.random.randint(0,2,[100,100]) # random mask
create_false_image(x,'/tmp/100x100size.tif')


class TestRaster(unittest.TestCase):
    def test_convert_datatype(self):
        
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
        assert(processing.convert_dt(gdal.GDT_Int16,to_otb_dt=True) == 'int16')
        assert(processing.convert_dt(np.dtype('float64').name,to_otb_dt=True) == 'double')
        
        assert(processing._convert_gdal_to_otb_dt(18) == 'cdouble') # if unknow, put cdouble (highest type)
        
    def _assert_np_gdt(self,in_conv,out_dt):
        assert(processing.convert_dt(in_conv)==out_dt)
        
        
    def test_gdt_minmax_values(self):
        assert(gdal.GDT_UInt16 == processing.get_gdt_from_minmax_values(500))
        assert(gdal.GDT_UInt32 == processing.get_gdt_from_minmax_values(max_value=155500))
        assert(gdal.GDT_Int32 == processing.get_gdt_from_minmax_values(max_value=0,min_value=-75500))
        assert(gdal.GDT_Int16 == processing.get_gdt_from_minmax_values(max_value=1,min_value=-5))
        assert(gdal.GDT_Float32 == processing.get_gdt_from_minmax_values(max_value=2,min_value=-55.55))
        assert(gdal.GDT_Byte == processing.get_gdt_from_minmax_values(max_value=222))
        assert(gdal.GDT_Float64 == processing.get_gdt_from_minmax_values(max_value =888E+40))
        assert(gdal.GDT_Float64 == processing.get_gdt_from_minmax_values(max_value=5,min_value = -888E+40))
        
    def test_rasterize(self):
        for invert in [True,False]:
            for field in ['class',False]:
                mem = processing.rasterize(raster,vector,field,out_image='MEM',invert=invert)
                assert(mem.RasterCount == 1)
                assert(mem.RasterXSize == rM.n_columns)
                assert(mem.RasterYSize == rM.n_lines)
            
    def test_noImg(self)    :    
        
        self.assertRaises(ReferenceError,processing.RasterMath,'None',verbose=0)
        
    def test_dimension(self)    :    
        assert(rM.n_bands == gdal.Open(raster).RasterCount)
        assert(rM.n_lines == gdal.Open(raster).RasterYSize)
        assert(rM.n_columns == gdal.Open(raster).RasterXSize)
        
    
    def test_readPerBand(self):
        for is_3d in [True,False]:
            rM_band = processing.RasterMath(raster,return_3d=is_3d,in_image_mask=mask)
            for idx,band in enumerate(rM_band.read_band_per_band()):
                print(band.ndim)
                if is_3d is True:
                    assert(band.ndim == 2)
                else:
                    assert(band.ndim == 2)
            del rM_band
    
    def test_3d(self)            :
        rM_3d = processing.RasterMath(raster,return_3d=True)
        self.assertRaises(ValueError,rM_3d.get_block,100)
        assert(rM_3d.get_random_block().ndim == 3)
        for block in rM.read_block_per_block():
            pass
        for band in rM.read_band_per_band():
            pass
        rM.custom_block_size(128,256)
        assert(rM.y_block_size==256)
        assert(rM.x_block_size==128)
        
        rM.custom_block_size(-1,-1)
        assert(rM.x_block_size == rM.n_columns)
        assert(rM.y_block_size == rM.n_lines)
        rM.custom_block_size(1/2,1/3)
        assert(rM.x_block_size == np.ceil(1/2*rM.n_columns))
        assert(rM.y_block_size == np.ceil(1/3*rM.n_lines))
        
        rM.add_image(raster)
        self.assertRaises(ValueError,rM.add_image,'/tmp/100x100size.tif')
        return_x = lambda x : x[0].astype(np.int16)
        rM.add_function(return_x,'/tmp/test_double.tif')
        rM.run()
        os.remove('/tmp/test_double.tif')
        assert(np.all(rM.get_random_block(random_state=12))== np.all(rM.get_random_block(random_state=12)))
        
        
    def test_mask(self)            :
        for is_3d in [True, False]:
            mask = '/tmp/mask.tif'
            processing.image_mask_from_vector(vector,raster,mask)
            mask_src = gdal.Open(mask)
            raster_src = gdal.Open(raster)
            mask_proj =osr.SpatialReference(wkt=mask_src.GetProjection())
            raster_proj = osr.SpatialReference(wkt=raster_src.GetProjection())
            assert(raster_proj.GetAttrValue('projcs') == mask_proj.GetAttrValue('projcs'))
            assert(mask_src.RasterCount == 1)
            assert(mask_src.RasterXSize == raster_src.RasterXSize)
            assert(mask_src.RasterYSize == raster_src.RasterYSize)
            rM_band = processing.RasterMath(raster,return_3d=is_3d)
            for idx,band in enumerate(rM_band.read_band_per_band()):
                pass
            rM_band.add_function(np.mean,axis=1,out_image='/tmp/mean.tif')
            rM_band.run()
            
            self.assertRaises(MemoryError,rM_band.run,'1K')
            
            assert(idx+1 == rM_band.n_bands)                        
            x = rM_band.get_random_block()
            assert(x.ndim == is_3d+2)
            os.remove('/tmp/mean.tif')
        
    
    def test_XYextraction(self):
        X = processing.extract_ROI(raster,vector,prefer_memory=False)
        
        
        self.assertRaises(ValueError,processing.extract_ROI,raster,vector,'Type')
        self.assertRaises(Exception,processing.extract_ROI,raster,vector,'no_field')
        
        assert(X.ndim == 2)
        
        X,y = processing.extract_ROI(raster,vector,'Class')
        assert(X.shape[0] == y.shape[0])
        
        X,y,g = processing.extract_ROI(raster,vector,'Class','uniquefid')
        assert(X.shape[0] == y.shape[0] == g.shape[0])
    
        self.assertRaises(ValueError,processing.extract_ROI,'wrong/path','wrong/path/too')
        assert(processing.extract_ROI(raster,vector).shape[1] == gdal.Open(raster).RasterCount)
        self.assertRaises(ValueError,processing.extract_ROI,raster,vector,'kodidk')
        
        
        
    def test_raster_math_mean(self):
        for is_3d in [True,False]:
            rM = processing.RasterMath(raster,return_3d = is_3d,verbose=is_3d,in_image_mask=mask,n_jobs=is_3d+1)
            if is_3d is False:
                # test without compression with reading/writing pixel per pixel, very slow...
                rM.custom_block_size(10,10) # to have full masked block
                rM.add_function(np.mean,'/tmp/mean.tif',axis=1,dtype=np.int16)
                rM.run()
            else:
                # test using default block size and high compressio of raster
                rM.add_function(np.mean,'/tmp/mean.tif',axis=1,dtype=np.int16,compress='high')
                rM.run()
            assert(gdal.Open('/tmp/mean.tif').RasterCount == 1)
            assert(gdal.Open('/tmp/mean.tif').RasterXSize == rM.n_columns)
            assert(gdal.Open('/tmp/mean.tif').RasterYSize == rM.n_lines)
            
            os.remove('/tmp/mean.tif')
            
    def test_unknow_fields(self):
        self.assertRaises(ValueError,processing.extract_ROI,raster,vector,'wrong_field')
        self.assertRaises(ValueError,processing.read_vector_values,vector)
        self.assertRaises(Exception,processing.read_vector_values,'wrong_path')
        self.assertRaises(ValueError,processing.read_vector_values,vector,'wrong_field')
        self.assertRaises(ValueError,processing.read_vector_values,vector,band_prefix='wrong_field')
        self.assertRaises(ReferenceError,processing.RasterMath,raster,in_image_mask='kiki')
    
    def test_addfid(self):
        copyfile(vector,'/tmp/test.gpkg')
        for tf in [True,False]:
            processing._add_vector_unique_fid('/tmp/test.gpkg',unique_field='to_create',verbose=tf)
        processing.sample_extraction(raster,'/tmp/test.gpkg','/tmp/test_roi.gpkg',band_prefix='band',verbose=1)
        self.assertRaises(Warning,processing.sample_extraction,raster,'/tmp/test.gpkg','/test/vector.ppkg')
        os.remove('/tmp/test.gpkg')
        
        y_ = processing.read_vector_values('/tmp/test_roi.gpkg',band_prefix='band',verbose=1)
        assert(y_.shape[1] == gdal.Open(raster).RasterCount)
        os.remove('/tmp/test_roi.gpkg')
     
    def test_centroid(self):
     
         Xc,yc = load_historical_data(centroid=True,return_X_y=True)
         Xc_file, yc_file= load_historical_data(centroid=True)
         assert(os.path.exists(Xc_file))
         assert(os.path.exists(yc_file))
         assert(Xc.shape[0] == processing.read_vector_values(vector,'Type').shape[0])
         
    def test_extract_position(self):
        X,pixel_position=processing.extract_ROI(raster,vector,get_pixel_position=True,prefer_memory=False)
        assert(pixel_position.shape[0] == X.shape[0])
        
    def test_get_parameter(self):
        rM = processing.RasterMath(raster)
        assert(isinstance(rM.get_raster_parameters(),dict))
        rM.custom_raster_parameters(['TILED=NO'])
        assert(rM.get_raster_parameters() == ['TILED=NO'])
    
    def test_get_distance_matrix(self):
        distance_matrix,label = processing.get_distance_matrix(raster,vector,'Class')
        assert(label.size== distance_matrix.shape[0])
        
if __name__ == "__main__":
    unittest.main()
    