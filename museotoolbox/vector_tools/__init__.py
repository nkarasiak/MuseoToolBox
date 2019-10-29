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
# @git:     www.github.com/nkarasiak/MuseoToolBox
# =============================================================================
"""
The :mod:`museotoolbox.vector_tools` module gathers vector functions.
"""
import os
from osgeo import ogr
import numpy as np

from ..raster_tools import getSamplesFromROI, sampleExtraction, rasterMaskFromVector

from ..internal_tools import progressBar


def getDistanceMatrix(inRaster, inVector, inLevel=False, verbose=False):
    """
    Return for each pixel, the distance one-to-one to the other pixels listed in the vector.

    Parameters
    ----------
    inRaster : str
        Path of the raster file.
    inVector : str
        Path of the vector file.

    Returns
    --------
    distanceMatrix : array of shape (nSamples,nSamples)
    """
    if inLevel is not False:
        onlyCoords = False
    else:
        onlyCoords = True

    coords = getSamplesFromROI(
        inRaster, inVector, inLevel, getCoords=True, onlyCoords=onlyCoords, verbose=verbose)
    from scipy.spatial import distance
    if inLevel:
        inLabel = coords[1]
        coords = coords[2]

    distanceMatrix = np.asarray(distance.cdist(
        coords, coords, 'euclidean'), dtype=np.uint64)

    if inLevel:
        return distanceMatrix, inLabel
    else:
        return distanceMatrix


def _getOgrDataTypeToNumpy(ogrType):
    FIELD_TYPES = [
        np.int32,          # OFTInteger, Simple 32bit integer
        None,           # OFTIntegerList, List of 32bit integers
        np.float64,       # OFTReal, Double Precision floating point
        None,           # OFTRealList, List of doubles
        np.str,          # OFTString, String of ASCII chars
        None,           # OFTStringList, Array of strings
        None,           # OFTWideString, deprecated
        None,           # OFTWideStringList, deprecated
        None,           # OFTBinary, Raw Binary data
        None,           # OFTDate, Date
        None,           # OFTTime, Time
        None,           # OFTDateTime, Date and Time
    ]
    numpyDT = FIELD_TYPES(ogrType)
    if numpyDT is None and ogrType > 4:
        numpyDT = np.str

    return numpyDT


def getDriverAccordingToFileName(fileName):
    """
    Return driver name used in OGR accoriding to the extension of the vector.

    Parameters
    ----------
    fileName : str.
        Path of the vector with extension.

    Returns
    -------
    driverName : str
        'SQLITE', 'GPKG', 'ESRI Shapefile'...

    Examples
    --------
    >>> mtb.vector_tools.getDriverAccordingToFileName('goVegan.gpkg')
    'GPKG'
    >>> mtb.vector_tools.getDriverAccordingToFileName('stopEatingAnimals.shp')
    'ESRI Shapefile'
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
    Read values from vector. Will list all fields beginning with the roiprefix 'band-' for example.

    Parameters
    ----------
    vector : str
        Vector path ('myFolder/class.shp',str).
    *args : str
        Field name containing the field to extract values from (i.e. 'class', str).
    **kwargs : arg
        - bandPrefix = 'band-' which is the common suffix listing the spectral values (i.e. bandPrefix = 'band-').
        - getFeatures = True, will return features in one list AND spatial Reference.

    Returns
    -------
    List values, same length as number of parameters.
    If bandPrefix as parameters, will return one array with n dimension.

    See also
    ---------
    museotoolbox.raster_tools.getSamplesFromROI : extract raster values from vector file.

    Examples
    ---------
    >>> from museotoolbox.datasets import getHistoricalMap
    >>> _,vector=getHistoricalMap()
    >>> Y = readValuesFromVector(vector,'Class')
    array([1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 5, 4, 5, 3, 3, 3], dtype=int32)
    >>> Y,fid = readValuesFromVector(vector,'Class','uniquefid')
    (array([1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 5, 4, 5, 3, 3, 3], dtype=int32),
     array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=int32))
    """

    try:
        file = ogr.Open(vector)
        lyr = file.GetLayer()
    except BaseException:
        raise Exception("Can't open {} file".format(vector))

    # get all fields and save only roiFields
    ldefn = lyr.GetLayerDefn()
    listFields = []

    # add kwargs
    extractBands = False
    getFeatures = False
    if kwargs:
        # check if need to extract bands from vector
        if 'bandPrefix' in kwargs.keys():
            extractBands = True
            bandPrefix = kwargs['bandPrefix']
        # check if need to extract features from vector
        if 'getFeatures' in kwargs.keys():
            getFeatures = kwargs['getFeatures']

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

    if len(kwargs) == 0 and len(args) == 0:
        print('These fields are available : {}'.format(listFields))
    else:

        if extractBands and len(bandsFields) == 0:
            raise ValueError(
                'Band prefix field "{}" do not exists. These fields are available : {}'.format(
                    bandPrefix, listFields))

        # Initialize empty arrays
        if len(args) > 0:  # for single fields
            ROIlevels = [np.zeros(lyr.GetFeatureCount()) for i in args]

        if extractBands:  # for bandPrefix
            ROIvalues = np.zeros(
                [lyr.GetFeatureCount(), len(bandsFields)], dtype=np.int32)

        # Listing each feature and store to array
        for i, feature in enumerate(lyr):
            if extractBands:
                for j, band in enumerate(bandsFields):
                    feat = feature.GetField(band)
                    if i == 0:
                        ROIvalues.astype(type(feat))

                    ROIvalues[i, j] = feat
            if len(args) > 0:
                try:
                    for a in range(len(args)):
                        feat = feature.GetField(args[a])
                        if i == 0:
                            ROIlevels[a] = ROIlevels[a].astype(type(feat))
                        ROIlevels[a][i] = feature.GetField(args[a])
                except BaseException:
                    raise ValueError(
                        "Field \"{}\" do not exists. These fields are available : {}".format(
                            args[a], listFields))
            if getFeatures:
                features.append(feature)

        # Initialize return
        fieldsToReturn = []

        # if bandPrefix
        if extractBands:
            fieldsToReturn.append(ROIvalues)

        # if single fields
        if len(args) > 0:
            for i in range(len(args)):
                fieldsToReturn.append(ROIlevels[i])

        # if features
        if getFeatures:
            fieldsToReturn.append(features)
            fieldsToReturn.append(srs)
        # if 1d, to turn single array
        if len(fieldsToReturn) == 1:
            fieldsToReturn = fieldsToReturn[0]

        return fieldsToReturn


def addUniqueIDForVector(inVector, uniqueField='uniquefid', verbose=True):
    """
    Add a field in the vector with an unique value
    for each of the feature.

    Parameters
    -----------
    inVector : str
        Path of the vector file.
    uniqueField : str
        Name of the field to create
    verbose : bool or int, default True.
    Returns
    --------
    None

    Examples
    ---------
    >>> addUniqueIDForVector('myDB.gpkg',uniqueField='polygonid')
    Adding polygonid [########################################]100%
    """
    pB = progressBar(100, message='Adding ' + uniqueField)

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
        if verbose > 0:
            print(
                'Field \'{}\' is already in {}'.format(
                    uniqueField, inVector))
        inSrc.Destroy()
    else:
        newField = ogr.FieldDefn(uniqueField, ogr.OFTInteger)
        newField.SetWidth(20)
        inLyr.CreateField(newField)

        FIDs = [feat.GetFID() for feat in inLyr]

        ThisID = 1

        for idx, FID in enumerate(FIDs):
            pB.addPosition(idx / len(FIDs) + 1 * 100)
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
