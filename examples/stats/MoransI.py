# -*- coding: utf-8 -*-
"""
Compute Moran's I with different lags from raster
===============================================================

Compute Moran's I with different lags, support mask.

"""

##############################################################################
# Import librairies
# -------------------------------------------
import numpy as np
from museotoolbox.stats import Moran
from matplotlib import pyplot as plt
import gdal,osr

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------
raster = '/tmp/autocorrelated_moran.tif'
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
create_false_image(x,raster)
plt.imshow(x)

################################
# Compute Moran's I for lag 1
lags =  [1,3,5]
Is = []
for lag in lags:
    MoransI = Moran(raster,lag=lag)
    Is.append(MoransI.scores['I'])

#######################
# Plot result
# -----------------------------------
from matplotlib import pyplot as plt 
plt.title('Evolution of Moran\'s I')
plt.plot(lags,np.mean(Is,axis=1),'-o')
plt.xlabel('Spatial lag')
plt.xticks(lags)
plt.ylabel('Moran\'s I')