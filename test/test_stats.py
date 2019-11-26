# -*- coding: utf-8 -*-
import unittest

import numpy as np
from museotoolbox import stats
import gdal
import osr
import os
confusion_matrix = np.array([[5,1],[2,2]])

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

class TestStats(unittest.TestCase):
    def test_Moran(self):
        self.assertRaises(ReferenceError,stats.Moran,in_image='N/A')
        x = np.zeros((100,100),dtype=int)
        # max autocorr
        x[:50,:] = 1
        x[50:,:] = 0
        create_false_image(x,'/tmp/test_moran.tif')
        moran = stats.Moran('/tmp/test_moran.tif',lag=1)
        assert(0.95 <= np.round(moran.I,0))
        os.remove('/tmp/test_moran.tif')

    def test_comm_om(self):
        comm_om = stats.commission_omission(confusion_matrix)

        assert(comm_om[0] == [confusion_matrix[0,1]/np.sum(confusion_matrix[0,:])*100,confusion_matrix[1,0]/np.sum(confusion_matrix[1,:])*100])
        assert(comm_om[1] == [confusion_matrix[1,0]/np.sum(confusion_matrix[:,0])*100,confusion_matrix[0,1]/np.sum(confusion_matrix[:,1])*100])
            
        sts = stats.ComputeConfusionMatrix(yp=np.asarray([1,1,1,1,1,1,2,2,2,2]),yr=np.asarray([1,1,1,1,1,2,1,1,2,2]))
        assert(np.all(confusion_matrix == sts.confusion_matrix))

    def stats_from_cm(self):
        
        sts = stats.ConfusionMatrixStats(confusion_matrix)
        assert(sts.OA  == (np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)))
        assert(sts.n == np.sum(confusion_matrix))
    
if __name__ == "__main__":
    unittest.main()
    