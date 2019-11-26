# -*- coding: utf-8 -*-
"""
Spatial Leave-Aside-Out (SLAO)
======================================================

This example shows how to make a Spatial Leave-Aside-Out.

See https://doi.org/10.1016/j.foreco.2013.07.059

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.cross_validation import SpatialLeaveAsideOut
from museotoolbox import datasets,geo_tools

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.load_historical_data(low_res=True)
field = 'Class'
X,y = geo_tools.extract_ROI(raster,vector,field)
distance_matrix = geo_tools.get_distance_matrix(raster,vector)

##############################################################################
# Create CV
# -------------------------------------------
# n_splits will be the number  of the least populated class

SLOPO = SpatialLeaveAsideOut(valid_size=1/3,n_splits=2,
                             distance_matrix=distance_matrix,random_state=2)

print(SLOPO.get_n_splits(X,y))

###############################################################################
# .. note::
#    Split is made to generate each fold

for tr,vl in SLOPO.split(X,y):
    print(tr.shape,vl.shape)  

###############################################################################
#    Save each train/valid fold in a file
# -------------------------------------------
# In order to translate polygons into points (each points is a pixel in the raster)
# we use sampleExtraction from vector_tools to generate a temporary vector.

geo_tools.sample_extraction(raster,vector,out_vector='/tmp/pixels.gpkg',verbose=False)
trvl = SLOPO.save_to_vector('/tmp/pixels.gpkg',field,out_vector='/tmp/SLOPO.gpkg')
for tr,vl in trvl:
    print(tr,vl)
 
    
###############################################################################
#    Plot example on how a polygon was splitted

import ogr
import numpy as np    
from matplotlib import pyplot as plt
# Read all features in layer and store as paths
xyl= np.array([],dtype=float).reshape((-1,3))
for idx,vector in enumerate([tr,vl]):
    ds = ogr.Open(vector)
    lyr = ds.GetLayer(0)
    lyr.SetAttributeFilter ( "uniquefid=17" ) # select a specific group
    for feat in lyr:
        geom = feat.GetGeometryRef()
        xyl = np.vstack((xyl,np.asarray((geom.GetX(),geom.GetY(),idx))))
    
trPoints = xyl[xyl[:,2]==0][:,:2]
vlPoints = xyl[xyl[:,2]==1][:,:2]
plt.scatter(trPoints[:,0],trPoints[:,1],label='train',color='C0')
plt.scatter(vlPoints[:,0],vlPoints[:,1],label='valid',color='C1')
plt.legend()
plt.show()