# -*- coding: utf-8 -*-
"""
Leave One Out Per Class (LOOPC)
======================================================

This example shows how to make a Leave One Out for each class.

"""

##############################################################################
# Import librairies
# -------------------------------------------

from museotoolbox.cross_validation import LeaveOneOut
from museotoolbox import datasets

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

X,y = datasets.historicalMap(return_X_y=True)

##############################################################################
# Create CV
# -------------------------------------------
LOOPC = LeaveOneOut(random_state=8,verbose=False)
for tr,vl in LOOPC.split(X=None,y=y):
    print(tr,vl)

###############################################################################
# .. note::
#    Split is made to generate each fold

# Show label

for tr,vl in LOOPC.split(X=None,y=y):
    print(y[vl])
    
###############################################################################
#    Save each train/valid fold in a file
# -------------------------------------------
# In order to translate polygons into points (each points is a pixel in the raster)
# we use sampleExtraction from vector_tools to generate a temporary vector.

trvl = LOOPC.saveVectorFiles(datasets.historicalMap()[1],'Class',outVector='/tmp/LOO.gpkg')
for tr,vl in trvl:
    print(tr,vl)
 
###############################################################################
#    Plot example on how a polygon was splitted

import ogr
import numpy as np    
from matplotlib import pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

# Prepare figure
plt.ioff()
ax=plt.subplot(1,1,1)
ax = plt.gca()


xBounds,yBounds=[[],[]]

for idx,vector in enumerate([tr,vl]):
    # Read all features in layer and store as paths    
    ds = ogr.Open(vector)
    lyr = ds.GetLayer(0)
    
    for feat in lyr:
        paths = []
        codes = []
        all_x = []
        all_y = []
        
        for geom in feat.GetGeometryRef():
            x = [geom.GetX(j) for j in range(geom.GetPointCount())]
            y = [geom.GetY(j) for j in range(geom.GetPointCount())]
            print(y)
            codes += [mpath.Path.MOVETO] + \
                             (len(x)-1)*[mpath.Path.LINETO]
            all_x += x
            all_y += y
        path = mpath.Path(np.column_stack((all_x,all_y)), codes)
        paths.append(path)
                
        # Add paths as patches to axes
        for path in paths:
            if idx==0:
                ax.add_patch(mpatches.PathPatch(path,color='C0'))
            else:
                ax.add_patch(mpatches.PathPatch(path,color='C1'))
                
        xBounds.append([np.min(all_x),np.max(all_x)])
        yBounds.append([np.min(all_y),np.max(all_y)])
       

ax.set_xlim(np.min(np.array(xBounds)[:,0]),np.max(np.array(xBounds)[:,1]))
ax.set_ylim(np.min(np.array(yBounds)[:,0]),np.max(np.array(yBounds)[:,1]))


legend = [mpatches.Patch(color='C0', label='Train'),mpatches.Patch(color='C1', label='Valid')]
plt.legend(handles=legend)

plt.show()
