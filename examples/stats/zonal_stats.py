# -*- coding: utf-8 -*-
"""
Compute quality index from confusion matrix
===============================================================

Compute different quality index  (OA, Kappa and F1) directly
from confusion matrix.

"""

##############################################################################
# Import librairies
# -------------------------------------------
import numpy as np
from museotoolbox.stats import zonal_stats
from museotoolbox.datasets import load_historical_data

##############################################################################
# Load dataset
# -------------------------------------------

raster,vector = load_historical_data()

##############################################################################
# Compute mean and variance per polygon
# ----------------------------------------------------
mean,var = zonal_stats(raster,vector,'uniquefid',stats=['mean','var'])
print(mean.shape)

#####################################
# Show mean value
print('For polygon 1 : ')
for band_idx,band in enumerate(['blue','green','red']):
    print('Mean value in {} band is : {}'.format(band,mean[0,band_idx]))

#####################################
# Show variance value    
print('For polygon 1 : ')
for band_idx,band in enumerate(['blue','green','red']):
    print('Variance value in {} band is : {}'.format(band,var[0,band_idx]))
    
###############################################"
# You can put in stats, every numpy function
#
# For example here : mean, median, max, min
    
mean,median,amax,amin = zonal_stats(raster,vector,'uniquefid',stats=['mean','median','max','min'])

print('For polygon 1 : ')
for band_idx,band in enumerate(['blue','green','red']):
    print('Min value in {} band is : {}'.format(band,amin[0,band_idx]))
    print('Max value in {} band is : {}'.format(band,amax[0,band_idx]))

