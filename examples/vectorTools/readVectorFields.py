# -*- coding: utf-8 -*-
"""
Read fields from vector
======================================================

This example shows how to read fields values from
a vector file."""

##############################################################################
# Import librairies
# -------------------

import MuseoToolBox as mtb

##############################################################################
# Load HistoricalMap dataset
# ----------------------------

raster,vector = mtb.datasets.getHistoricalMap()

###############################################################################
# .. note::
#    If you have no memories on what the fields name are, simply put the vector path

mtb.vectorTools.readValuesFromVector(vector)

#############################################
# Read values from field 'Class'
# --------------------------------

Y = mtb.vectorTools.readValuesFromVector(vector,'Class')
print(Y)
print(Y.shape)

#############################################
# Read values from field beginning with 'C'
# -------------------------------------------
# As multiple fields can begin with C, function returns a column per field

C = mtb.vectorTools.readValuesFromVector(vector,bandPrefix='C')
print(C)
print(C.shape)
