    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:38:05 2019

@author: nicolas

in_image : string.
    A filename or path of a raster file.
    It could be any file that GDAL can open.

in_vector : string.
    A filename or path corresponding to a vector file.
    It could be any file that GDAL/OGR can open.
    
out_image : string.
    A geotiff extension filename corresponding to a raster image to create.
    
X : array-like, shape = [n_samples, n_features]
    The training input samples.

y : array-like, shape = [n_samples]
    The target values.
    
in_image_mask : str
    A filename or path corresponding to a raster image.
    0 values are considered as masked data.

out_image : str
    A filename or path corresponding to a geotiff (.tif) raster image to save.
    0 values are considered as masked data.


"""


