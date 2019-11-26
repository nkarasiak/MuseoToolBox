# -*- coding: utf-8 -*-
import unittest

import numpy as np
from museotoolbox import stats
import gdal
import osr
import os
confusion_matrix = np.array([[5,1],[2,2]])
yp = [1,1,1,1,1,1,2,2,2,2]
yr = [1,1,1,1,1,2,1,1,2,2]


        
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
x[50:,:] = 0
create_false_image(x,'/tmp/test_moran.tif')
        
class TestStats(unittest.TestCase):
    def test_Moran_param(self):
        m = stats.Moran('/tmp/test_moran.tif',lag=[1,2])
        assert(m.get_n_neighbors(x[:3,:3],x[:3,:3],weights=x[:3,:3]) == 8)
        m.lags == [1,2]
        assert(len(m.scores['I']) == len(m.lags))
        
    def test_Moran(self):
        self.assertRaises(ReferenceError,stats.Moran,in_image='N/A')

        moran = stats.Moran('/tmp/test_moran.tif',lag=1)
        assert(0.95 <= np.round(moran.I,0))

    def test_comm_om(self):
        comm_om = stats.commission_omission(confusion_matrix)

        assert(comm_om[0] == [confusion_matrix[0,1]/np.sum(confusion_matrix[0,:])*100,confusion_matrix[1,0]/np.sum(confusion_matrix[1,:])*100])
        assert(comm_om[1] == [confusion_matrix[1,0]/np.sum(confusion_matrix[:,0])*100,confusion_matrix[0,1]/np.sum(confusion_matrix[:,1])*100])
            
        sts = stats.ComputeConfusionMatrix(yp=np.asarray([1,1,1,1,1,1,2,2,2,2]),yr=np.asarray([1,1,1,1,1,2,1,1,2,2]),OA=True,  kappa=  True,F1=True)
        assert(np.all(confusion_matrix == sts.confusion_matrix))
        assert(len(sts.F1) == 2)

    def stats_from_cm(self):
        
        sts = stats.ConfusionMatrixStats(confusion_matrix)
        assert(sts.OA  == (np.sum(np.diag(confusion_matrix))/np.sum(confusion_matrix)))
        assert(sts.n == np.sum(confusion_matrix))
        
        sts_from_matrix = stats.ComputeConfusionMatrix(np.asarray(sts.yr),np.asarray(sts.yp),OA=True)
        assert(sts_from_matrix.OA == sts.OA)
        
if __name__ == "__main__":
    unittest.main()
    os.remove('/tmp/test_moran.tif')

    