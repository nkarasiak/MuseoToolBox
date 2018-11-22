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

from __future__ import absolute_import, print_function
from .. import rasterTools, vectorTools
from . import crossValidationClass
import os
import numpy as np


class _sampleSelection:
    def __init__(self):
        """
        sampleSelection generate the duo valid/train samples in order to your samplingMethods choosen function.

        Parameters
        ----------
        inVector : str or array.
            if str, path of the vector. If array, numpy array of labels.
        inField : str or None if inVector is np.ndarray.
            if str, field name from the vector.

        Returns
        ----------
        getCrossValidation() : Function.
            Get a memory cross validation to use directly in Scikit-Learn.

        saveVectorFiles() : Need default output name (str).
            To save as many vector files (train/valid) as your Cross Validation method outputs.

        getSupportedExtensions() : Function.
            Show you the list of supported vector extensions type when using saveVectorFiles function.

        reinitialize() : Function.
            If you need to regenerate the cross validation, you need to reinitialize it.

        """
        if isinstance(self.inVector, (np.ndarray, list)):
            self.inVectorIsArray = True
        else:
            self.inVectorIsArray = False

        self.extensions = ['sqlite', 'shp', 'netcdf', 'gpx', 'gpkg']
        self.driversName = [
            'SQLITE',
            'ESRI Shapefile',
            'netCDF',
            'GPX',
            'GPKG']

        self.__alertMessage = 'It seems you already generated the cross validation. \n Please use reinitialize function if you want to regenerate the cross validation. \n But check if you defined a seed number in your samplingMethods function to have the same output.'
        self.__alreadyRead = False

        # create unique random state if no seed
        if self.params['seed'] is None:
            self.seedGenerated = True
            import time
            self.params['seed'] = int(time.time())
        # Totally random
        if self.samplingType == 'random':
            if self.inVectorIsArray:
                FIDs = self.inVector
                if isinstance(FIDs, list):
                    FIDs = np.array(FIDs)
            else:
                FIDs, self.fts, self.srs = vectorTools.readValuesFromVector(
                    self.inVector, self.inField, getFeatures=True, verbose=self.verbose)
            FIDs = FIDs.flatten()

            self.crossvalidation = crossValidationClass.randomPerClass(
                FIDs=FIDs, verbose=self.verbose, **self.params)

        if self.samplingType == 'Spatial':
            self.__prepareDistanceCV()
            self.crossvalidation = crossValidationClass.distanceCV(
                Y=self.Y, verbose=self.verbose, **self.params)

        # For Stand Split
        if self.samplingType == 'Group':
            #FIDs,STDs,srs,fts = vectorTools.readFieldVector(inVector,inField,inStand,getFeatures=True)
            if self.inVectorIsArray:
                FIDs = self.inVector
                FIDs = FIDs.flatten()
                STDs = self.inGroup
            else:
                FIDs, STDs, self.fts, self.srs = vectorTools.readValuesFromVector(
                    self.inVector, self.inField, self.inGroup, getFeatures=True, verbose=self.verbose)
                FIDs = FIDs.flatten()
            self.crossvalidation = crossValidationClass.groupCV(
                FIDs,
                STDs,
                valid_size=self.params['valid_size'],
                n_splits=self.params['n_splits'],
                seed=self.params['seed'],
                verbose=self.verbose)

    def __prepareDistanceCV(self):
        # Split at maximum distance beyond each point
        # For Spatial-Leave-One-Out
        if not self.inVectorIsArray:
            self.FIDs, self.fts, self.srs = vectorTools.readValuesFromVector(
                self.inVector, self.inField, getFeatures=True, verbose=self.verbose)
            if hasattr(self, 'inGroup'):
                X, self.Y, inGroup = rasterTools.getSamplesFromROI(
                    self.inRaster, self.inVector, self.inField, self.inGroup, verbose=self.verbose)
                self.params['group'] = inGroup
            else:
                X, self.Y = rasterTools.getSamplesFromROI(
                    self.inRaster, self.inVector, self.inField, verbose=self.verbose)

            if self.FIDs.shape[0] != self.Y.shape[0]:
                self.SLOOnotSamesize = True
                self.errorSLOOmsg = 'Number of features if different of number of pixels. Please use rasterTools.sampleExtraction if you want to save as vector the Cross-Validation.'
                print(self.errorSLOOmsg)
            else:
                self.SLOOnotSamesize = False
        else:
            if hasattr(self, 'inGroup'):
                self.params['group'] = self.inGroup
            self.Y = self.inVector

    def reinitialize(self):
        _sampleSelection.__init__(self)

    def getSupportedExtensions(self):
        print('Museo ToolBox supported extensions are : ')
        for idx, ext in enumerate(self.extensions):
            print(3 * ' ' + '- ' + self.driversName[idx] + ' : ' + ext)

    def split(self, **sklearnCompatiblity):
        if self.__alreadyRead:
            self.reinitialize()
        self.__alreadyRead = True
        return self.crossvalidation

    def getCrossValidation(self):
        if self.__alreadyRead:
            self.reinitialize()
        self.__alreadyRead = True
        return self.crossvalidation

    def saveVectorFiles(self, outVector):
        if self.inVectorIsArray:
            raise Exception(
                'To save vector files, you need to use in input a vector, not an array')
        if self.samplingType == 'SLOO':
            if self.SLOOnotSamesize:
                raise Exception(self.errorSLOOmsg)
        self.__fileName, self.__ext = os.path.splitext(outVector)

        if self.__ext[1:] not in self.extensions:
            print(
                'Your extension {} is not recognized as a valid extension for saving shape.'.format(
                    self.__ext))
            self.getSupportedExtensions()
            print('We recommend you to use sqlite extension.')

        else:
            if self.__alreadyRead:
                self.reinitialize()
            listOutput = []
            self.cv = []
            for idx, trvl in enumerate(self.crossvalidation):
                self.cv.append([trvl[0], trvl[1]])
                trFeat = [self.fts[int(i)] for i in trvl[0]]
                vlFeat = [self.fts[int(i)] for i in trvl[1]]
                tr = self.__fileName + '_train_' + str(idx) + self.__ext
                vl = self.__fileName + '_valid_' + str(idx) + self.__ext
                self.__saveToShape__(trFeat, self.srs, tr)
                self.__saveToShape__(vlFeat, self.srs, vl)
                listOutput.append([tr, vl])
            self.__alreadyRead = True
            return listOutput

    def __saveToShape__(self, array, srs, outShapeFile):
        # Parse a delimited text file of volcano data and create a shapefile
        # use a dictionary reader so we can access by field name
        # set up the shapefile driver
        import ogr

        driverIdx = [x for x, i in enumerate(
            self.extensions) if i == self.__ext[1:]][0]
        outDriver = ogr.GetDriverByName(self.driversName[driverIdx])

        # create the data source
        if os.path.exists(outShapeFile):
            outDriver.DeleteDataSource(outShapeFile)
        # Remove output shapefile if it already exists

        # options = ['SPATIALITE=YES'])
        ds = outDriver.CreateDataSource(outShapeFile)

        # create the spatial reference, WGS84

        lyrout = ds.CreateLayer(
            'randomSubset',
            srs=srs,
            geom_type=ogr.wkbPoint)
        fields = [
            array[1].GetFieldDefnRef(i).GetName() for i in range(
                array[1].GetFieldCount())]
        if lyrout is None:
            raise Exception('Failed to create file ' + str(outShapeFile))

        if self.__ext[1:] != 'shp':
            isShp = False
            lyrout.StartTransaction()
        else:
            isShp = True

        for i, f in enumerate(fields):
            field_name = ogr.FieldDefn(
                f, array[1].GetFieldDefnRef(i).GetType())
            if isShp:
                field_name.SetWidth(array[1].GetFieldDefnRef(i).GetWidth())
            lyrout.CreateField(field_name)

        for k in array:
            lyrout.CreateFeature(k)
        k, i, f = None, None, None

        if not isShp:
            lyrout.CommitTransaction()
        # Save and close the data source
        ds = None
