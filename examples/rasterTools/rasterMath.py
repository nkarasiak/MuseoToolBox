# -*- coding: utf-8 -*-
"""
Basics to use rasterMath
===============================================================

Test rasterMath

"""

##############################################################################
# Import librairies
# -------------------------------------------

from MuseoToolBox.rasterTools import rasterMath
from MuseoToolBox import datasets

##############################################################################
# Load HistoricalMap dataset
# -------------------------------------------

raster,vector = datasets.getHistoricalMap()
field = 'Class'
group = 'uniquefid'

##############################################################################
# Initialize Random-Forest
# ---------------------------

rM = rasterMath(raster)

print(rM.getRandomBlock())
##########################
# Plot example
