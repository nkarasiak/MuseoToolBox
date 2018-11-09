#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# ___  ___                       _____           _______
# |  \/  |                      |_   _|         | | ___ \
# | .  . |_   _ ___  ___  ___     | | ___   ___ | | |_/ / _____  __
# | |\/| | | | / __|/ _ \/ _ \    | |/ _ \ / _ \| | ___ \/ _ \ \/ /
# | |  | | |_| \__ \  __/ (_) |   | | (_) | (_) | | |_/ / (_) >  <
# \_|  |_/\__,_|___/\___|\___/    \_/\___/ \___/|_\____/ \___/_/\_\
#
# @author:  Nicolas Karasiak
# @site:    www.karasiak.net
# @git:     www.github.com/lennepkade/MuseoToolBox
# =============================================================================
from __future__ import absolute_import,print_function
import os
from osgeo import ogr
import numpy as np

from .crossValidationSelection import samplingMethods, sampleSelection
from .sampleExtraction import sampleExtraction
from . import crossValidationClass


def getDriverAccordingToFileName(fileName):
    """
    Return driver name used in OGR accoriding to the extension of the vector.

    Parameters
    ----------
    fileName : str.
        Path of the vector with extension.

    Output
    ----------
    driverName : str
        'SQLIRE', 'GPKG', 'ESRI Shapefile'...
    """
    extensions = ['sqlite', 'shp', 'netcdf', 'gpx', 'gpkg']
    driversName = ['SQLITE', 'ESRI Shapefile', 'netCDF', 'GPX', 'GPKG']

    fileName, ext = os.path.splitext(fileName)

    if ext[1:] not in extensions:
        msg = 'Your extension {} is not recognized as a valid extension for saving shape.\n'.format(
            ext)
        msg = msg + 'Supported extensions are ' + str(driversName) + '\n'
        msg = msg + 'We recommend you to use \'sqlite\' extension.'
        raise Warning(msg)
    else:
        driverIdx = [x for x, i in enumerate(extensions) if i == ext[1:]][0]
        driverName = driversName[driverIdx]
        return driverName


def readValuesFromVector(vector, *args, **kwargs):
    """
    Read values from vector. Will list all fields beginning with the roiprefix ('band_' for example)

    Parameters
    ----------
    vector : str
        Vector path ('myFolder/class.shp',str).
    *args : str
        Field name containing the field to extract values from (i.e. 'class', str).
    **kwargs : arg
        bandPrefix = 'band_' which is the common suffix listing the spectral values (i.e. bandPrefix = 'band_').
        getFeatures = True, will return features in one list AND spatial Reference.

    Output
    ----------
    Arr.
    """

    file = ogr.Open(vector)
    lyr = file.GetLayer()

    # get all fields and save only roiFields
    ldefn = lyr.GetLayerDefn()
    listFields = []

    # add kwargs
    if kwargs:
        # check if need to extract bands from vector
        if 'bandPrefix' in kwargs.keys():
            extractBands = True
            bandPrefix = kwargs['bandPrefix']
        else:
            extractBands = False

        # check if need to extract features from vector
        if 'getFeatures' in kwargs.keys():
            getFeatures = kwargs['getFeatures']
        else:
            getFeatures = False
    else:
        extractBands = False
        getFeatures = False

    if extractBands:
        bandsFields = []

    # if getFeatures, save Spatial Reference and Features
    if getFeatures:
        srs = lyr.GetSpatialRef()
        features = []

    # List available fields
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        if fdefn.name is not listFields:
            listFields.append(fdefn.name)
        if extractBands:
            if fdefn.name.startswith(bandPrefix):
                bandsFields.append(fdefn.name)
    if extractBands:
        if len(bandsFields) == 0:
            raise ValueError(
                'Band prefix field "{}" do not exists. These fields are available : {}'.format(
                    bandPrefix, listFields))

    # Initialize empty arrays
    if len(args) > 0:  # for single fields
        ROIlevels = np.zeros(
            [lyr.GetFeatureCount(), len(args)], dtype=np.int32)

    if extractBands:  # for bandPrefix
        ROIvalues = np.zeros(
            [lyr.GetFeatureCount(), len(bandsFields)], dtype=np.int32)

    # Listing each feature and store to array
    for i, feature in enumerate(lyr):
        if extractBands:
            for j, band in enumerate(bandsFields):
                ROIvalues[i, j] = feature.GetField(band)
        if len(args) > 0:
            try:
                for a in range(len(args)):
                    ROIlevels[i, a] = feature.GetField(args[a])
            except BaseException:
                raise ValueError(
                    "Field \"{}\" do not exists or is not an integer/float field. These fields are available : {}".format(
                        args[a], listFields))
        if getFeatures:
            features.append(feature)

    # Initialize return
    fieldsToReturn = []

    # if single fields
    if len(args) > 0:
        for i in range(len(args)):
            fieldsToReturn.append(np.asarray(ROIlevels)[:, i])
    # if bandPrefix
    if extractBands:
        fieldsToReturn.append(ROIvalues)
    # if features
    if getFeatures:
        fieldsToReturn.append(features)
        fieldsToReturn.append(srs)
    # if 1d, to turn single array
    if len(fieldsToReturn) == 1:
        fieldsToReturn = fieldsToReturn[0]

    return fieldsToReturn


def addUniqueIDForVector(inVector, uniqueField='uniquefid'):
    inDriverName = getDriverAccordingToFileName(inVector)
    inDriver = ogr.GetDriverByName(inDriverName)
    inSrc = inDriver.Open(inVector, 1)  # 1 for writable
    inLyr = inSrc.GetLayer()       # get the layer for this datasource
    inLyrDefn = inLyr.GetLayerDefn()

    if inDriverName == 'SQLITE':
        inLyr.StartTransaction()

    listFields = []
    for n in range(inLyrDefn.GetFieldCount()):
        fdefn = inLyrDefn.GetFieldDefn(n)
        if fdefn.name is not listFields:
            listFields.append(fdefn.name)
    if uniqueField in listFields:
        print('Field \'{}\' is already in {}'.format(uniqueField, inVector))
        inSrc.Destroy()
    else:
        newField = ogr.FieldDefn(uniqueField, ogr.OFTInteger)
        newField.SetWidth(20)
        inLyr.CreateField(newField)

        FIDs = [feat.GetFID() for feat in inLyr]

        ThisID = 1

        for FID in FIDs:
            feat = inLyr.GetFeature(FID)
            #ThisID = int(feat.GetFGetFeature(feat))
            # Write the FID to the ID field
            feat.SetField(uniqueField, int(ThisID))
            inLyr.SetFeature(feat)              # update the feature
            # inLyr.CreateFeature(feat)
            ThisID += 1

        if inDriverName == 'SQLITE':
            inLyr.CommitTransaction()
        inSrc.Destroy()
    return inVector
