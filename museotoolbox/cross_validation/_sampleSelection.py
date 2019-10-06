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
import os
from .. import raster_tools, vector_tools
from . import crossValidationClass
import numpy as np
from itertools import tee


class _sampleSelection:
    def __init__(self):
        """
        sampleSelection generate the duo valid/train samples in order to your samplingMethods choosen function.

        Methods
        ---------
        split(X,y,g) : Function.
            Get a memory cross validation to use directly in Scikit-Learn.

        saveVectorFiles() : Need default output name (str).
            To save as many vector files (train/valid) as your Cross Validation method outputs.

        __getSupportedExtensions() : Function.
            Show you the list of supported vector extensions type when using saveVectorFiles function.

        reinitialize() : Function.
            If you need to regenerate the cross validation, you need to reinitialize it.
        """
        self.__extensions = ['sqlite', 'shp', 'netcdf', 'gpx', 'gpkg']
        self.__driversName = [
            'SQLITE',
            'ESRI Shapefile',
            'netCDF',
            'GPX',
            'GPKG']

        self.__alertMessage = 'It seems you already generated the cross validation. \n Please use reinitialize function if you want to regenerate the cross validation. \n But check if you defined a random_state number in your samplingMethods function to have the same output.'
        self.__alreadyRead = False

        # create unique random state if no random_state
        if self.params['random_state'] is None:
            import time
            self.params['random_state'] = int(time.time())

    def reinitialize(self):
        _sampleSelection.__init__(self)

    def __getSupportedExtensions(self):
        print('Museo ToolBox supported extensions are : ')
        for idx, ext in enumerate(self.__extensions):
            print(3 * ' ' + '- ' + self.__driversName[idx] + ' : ' + ext)

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features), optional
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Subgroup labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        n_splits : int
            The number of splits.
        """

        if y is not None:
            self.y = y

        if self.crossvalidation.__name__ == 'distanceCV':
            # TODO : Find a better way to get n_splits for distanceCV
            # As distance may differ from real n_splits, hard to not run the
            # whole thing
            n_splits = 0
            for tr, vl in self.crossvalidation(
                    X=X, y=y, groups=groups, verbose=self.verbose, **self.params):
                n_splits += 1
        else:
            n_splits = self.crossvalidation(
                X=X, y=y, groups=groups, verbose=self.verbose, **self.params).n_splits

        return n_splits

    def split(self, X=None, y=None, groups=None):
        """
        Split the vector/array according to y and groups.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features), optional
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, of length n_samples
            The target variable for supervised learning problems.
        groups : array-like, with shape (n_samples,), optional
            Subgroup labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        --------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        """
        if y is None:
            y = self.Y
        y = y.reshape(-1, 1)
        if self.__alreadyRead:
            self.reinitialize()
        self.__alreadyRead = True

        return self.crossvalidation(
            X=X, y=y, groups=groups, verbose=self.verbose, **self.params)

    def saveVectorFiles(self, vector, field, raster=None,
                        groupsField=None, outVector=None):
        """
        Save to vector files each fold from the cross-validation.

        Parameters
        -------------
        vector : str.
            Path where the vector is stored.
        field : str.
            Name of the field containing the label.
        groupsField : str, or None.
            Name of the field containing the group/subgroup (or None
        outVector : str.
            Path and filename to save the different results.

        Returns
        ----------
        listOfOutput : list
            List containing the number of folds * 2
            train + validation for each fold.

        """
        import ogr
        src = ogr.Open(vector)
        srcLyr = src.GetLayerByIndex()
        self.wkbType = srcLyr.GetGeomType()
        if self.wkbType != 1:
            print("""Warning : This function generates vector files according to your vector.
        The number of features may differ from the number of pixels used in classification.
        If you want to save every ROI pixels in the vector, please use vector_tools.sampleExtraction before.""")
        del src, srcLyr

        fileName, self.__ext = os.path.splitext(outVector)

        if self.__ext[1:] not in self.__extensions:
            print(
                'Your extension {} is not recognized as a valid extension for saving shape.'.format(self.__ext))
            self.__getSupportedExtensions()
            raise Exception('We recommend you to use sqlite/gpkg extension.')

        if groupsField is None:
            groups = None
            y, fts, srs = vector_tools.readValuesFromVector(
                vector, field, getFeatures=True, verbose=self.verbose)
        else:
            y, groups, fts, srs = vector_tools.readValuesFromVector(
                vector, field, groupsField, getFeatures=True, verbose=self.verbose)

        if self.__alreadyRead:
            self.reinitialize()
        listOutput = []
        self.cv = []
        for idx, trvl in enumerate(self.split(None, y, groups)):
            if trvl is not None:
                self.cv.append([trvl[0], trvl[1]])
                trFeat = [fts[int(i)] for i in trvl[0]]
                vlFeat = [fts[int(i)] for i in trvl[1]]
                tr = fileName + '_train_' + str(idx) + self.__ext
                vl = fileName + '_valid_' + str(idx) + self.__ext
                self.__saveToShape__(trFeat, srs, tr)
                self.__saveToShape__(vlFeat, srs, vl)
                listOutput.append([tr, vl])
        self.__alreadyRead = True
        return listOutput

    def __saveToShape__(self, array, srs, outShapeFile):
        # Parse a delimited text file of volcano data and create a shapefile
        # use a dictionary reader so we can access by field name
        # set up the shapefile driver
        import ogr

        driverIdx = [x for x, i in enumerate(
            self.__extensions) if i == self.__ext[1:]][0]
        outDriver = ogr.GetDriverByName(self.__driversName[driverIdx])

        # create the data source
        if os.path.exists(outShapeFile):
            outDriver.DeleteDataSource(outShapeFile)
        # Remove output shapefile if it already exists

        # options = ['SPATIALITE=YES'])
        ds = outDriver.CreateDataSource(outShapeFile)

        # create the spatial reference, WGS84

        lyrout = ds.CreateLayer(
            'cross_validation',
            srs=srs,
            geom_type=self.wkbType)
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
