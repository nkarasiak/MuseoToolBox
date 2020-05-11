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
import numpy as np
from osgeo import ogr
from .. import processing


class distanceCV:
    def __init__(
            self,
            X,
            y,
            distance_matrix,
            distance_thresold=False,
            valid_size=1,
            n_repeats=False,
            n_splits=False,
            verbose=False,
            random_state=False,
            groups=None,
            distance_label=False,
            LOO_same_size=False):
        """Compute train/validation array with Spatial distance analysis.
        Object stops when less effective class number is reached (45 loops if your least class contains 45 ROI).
        Parameters
        ----------
        X : array-like
            Matrix of values. As many row as the Y array.
        Y : array-like
            contain class for each ROI. Same effective as distance_matrix.
        distance_matrix : array
            Matrix distance
        distance_thresold : int, float or False, optional (default=False).
            Distance(same unit of your distance_matrix).
            If False, will split spatially the dataset.
        n_repeats : int or False, optional (default=False).
            False : as loop as min effective class
        n_splits : int or False, optional (default=False).
            Number of split per CV. Default is the size of the smallest class.
        valid_size : False or float value.
            If float, value from 0 to 1 (percent).
        groups : array
            contain class (like Y), e.g. able to do a SLOO per Stand if you put your stand number here.
        stat : str.
            Path where to save csv stat (label, n trains, and mean distance between training per class).
        LOO_same_size : bool, optional (default=False).
            Use the same training size as SLOO but use samples without taking into account the distance.

        Returns
        -------
        train : array
            List of Y selected ROI for train
        validation : array
            List of Y selected ROI for validation
        """
        self.name = 'SLOO'

        self.distance_matrix = distance_matrix
        self.distance_thresold = distance_thresold
        self.y = y
        self.iterPos = 0

        self.y_labels = np.unique(y, return_counts=True)[0]

        self.distance_label = distance_label

        self.valid_size = valid_size
        self.LOO_same_size = LOO_same_size
        self.groups = groups

        self.verbose = verbose

        self.random_state = random_state

        self.mask = np.ones(np.asarray(self.y).shape, dtype=bool)

        if self.groups is None:
            self.minEffectiveClass = min(
                [len(self.y[self.y == i]) for i in np.unique(self.y)])
        else:
            self.minEffectiveClass = min(
                [len(np.unique(groups[np.where(y == i)[0]])) for i in np.unique(self.y)])

        if n_splits:
            self.minEffectiveClass = n_splits

        if n_repeats and self.valid_size >= 1:
            self.n_repeats = self.minEffectiveClass * n_repeats

        elif self.valid_size >= 1:
            # TODO : run self.__next__() to get real n_repeats as it depends on distance
            # but if running self.__next__() here, iterator will be empty
            # after.
            self.n_repeats = self.minEffectiveClass
        elif isinstance(self.valid_size, float):
            self.n_repeats = int(1 / (self.valid_size))
        else:
            self.n_repeats = self.minEffectiveClass

    def __iter__(self):
        return self

    # python 3 compatibility

    def __next__(self):
        return self.next()

    def next(self):
        emptyTrain = False

        if self.iterPos < self.n_repeats:
            self.nTries = 0
            completeTrain = False
            while completeTrain is False:
                if self.nTries < 100:
                    emptyTrain = False
                    if self.verbose:
                        print(53 * '=')
                    validation, train = np.array([[], []], dtype=np.int64)
                    for C in self.y_labels:
                        # Y is True, where C is the unique class
                        CT = np.where(self.y == C)[0]

                        currentCT = np.logical_and(self.y == C, self.mask == 1)

                        if np.where(currentCT)[
                                0].shape[0] == 0:  # means no more ROI
                            if self.verbose > 1:
                                print(
                                    str(C) + ' has no more valid pixel.\nResetting label mask.')
                            self.mask[self.y == C] = 1
                            currentCT = np.logical_and(
                                self.y == C, self.mask == 1)

                        if np.where(currentCT)[
                                0].shape[0] == 0:  # means no more ROI
                            if self.verbose > 1:
                                print(
                                    str(C) + ' has no more valid pixel. Reusing samples.')
                            self.mask[self.y == C] = 1
                            currentCT = np.logical_and(
                                self.y == C, self.mask == 1)

                        np.random.seed(self.random_state)
                        self.ROI = np.random.permutation(
                            np.where(currentCT)[0])[0]
                        # When doing Leave-One-Out per subgroup
                        if self.groups is not None:
                            if self.verbose > 1:
                                print('ROI stand is ' +
                                      str(self.groups[self.ROI]) +
                                      ' for label ' +
                                      str(C))
                            standPos = np.argwhere(
                                self.groups[self.ROI] == self.distance_label)[0][0]
                            distanceROI = (self.distance_matrix[standPos, :])
                            tmpValid = np.where(self.groups == self.groups[self.ROI])[
                                0].astype(np.int64)

                            tmpTrainGroup = np.unique(self.distance_label[np.where(
                                distanceROI >= self.distance_thresold)[0]])
                            tmpTrainGroup = tmpTrainGroup[np.isin(
                                tmpTrainGroup, self.groups[CT])]
                            tmpTrain = np.where(np.in1d(self.groups, tmpTrainGroup))[
                                0].flatten()

                            if tmpTrain.shape[0] == 0:
                                emptyTrain = True
                        # When doing Leave-One-Out
                        else:

                            # get line of distance for specific ROI
                            distanceROI = (
                                self.distance_matrix[self.ROI, :])[CT]
                            if self.valid_size is False:
                                tmpValid = np.array(
                                    [self.ROI], dtype=np.int64)
                                tmpTrain = CT[distanceROI >
                                              self.distance_thresold]
                                if self.LOO_same_size is True:
                                    np.random.seed(self.random_state)
                                    tmpTrain = np.random.permutation(
                                        CT)[:len(tmpTrain)]

                            else:
                                if self.valid_size >= 1:
                                    nToCut = self.valid_size
                                else:
                                    nToCut = int(self.valid_size * len(CT))

                                if self.distance_thresold is False:
                                    distanceToCut = np.sort(distanceROI)[
                                        :nToCut][-1]
                                    tmpValid = CT[distanceROI <=
                                                  distanceToCut]

                                    tmpTrain = CT[distanceROI >
                                                  distanceToCut]

                                else:
                                    tmpValid = np.asarray([self.ROI])

                                    tmpTrain = CT[distanceROI >
                                                  self.distance_thresold]

                                if self.LOO_same_size is True:
                                    np.random.seed(self.random_state)

                            if tmpTrain.shape[0] == 0:
                                emptyTrain = True

                        del CT
                        if emptyTrain is True:
                            if self.verbose:
                                print(
                                    'No training for class ' +
                                    str(C) +
                                    '. Doing a new fold.')

                            self.mask[tmpValid] = 0

                        else:
                            #
                            validation = np.concatenate((validation, tmpValid))
                            train = np.concatenate((train, tmpTrain))

                    if self.verbose:
                        print('Validation samples : ' + str(len(validation)))
                        print('Training samples : ' + str(len(train)))

                    # Mask selected validation
                    self.random_state += 1
                    if emptyTrain is True:
                        completeTrain = False
                        self.nTries += 1

                    else:
                        self.iterPos += 1
                        self.mask[validation] = 0
                        return train, validation
                else:
                    raise ValueError(
                        'Error : Not enough samples using this distance/valid_size.')

        else:
            raise StopIteration()


class randomPerClass:
    """
    Random array according to Y.

    Parameters
    ----------
    X : None
    Y : arr, default None.
        Label for each feature.
    groups : arr, default None.
        Group for each feature. For sklearn compatibility.
    valid_size : False or int
        1 to do a Leave-One-Out.
    train_size : float (<1.0) or int (>1).
        Percentage to keep for training or integer.
    n_splits : False or int.
        If False, will be the number of samples of the smallest class.
    random_state : int.
        random_state for numpy.
    verbose : boolean or int.

    Returns
    -------
    train,validation : array of indices
    """

    def __init__(self, X=None, y=None, groups=None,
                 valid_size=0.5, train_size=None, n_repeats=False,
                 random_state=None, verbose=False):
        self.y = y
        self.valid_size = valid_size
        self.train_size = 1 - self.valid_size

        smallestClass = np.min(np.unique(y, return_counts=True)[1])

        if n_repeats is False:
            if self.valid_size >= 1:
                self.n_repeats = smallestClass
            else:
                self.n_repeats = int(1 / self.valid_size)
        else:
            self.n_repeats = n_repeats

        if self.valid_size < 1:
            test_n_splits = int(valid_size * smallestClass)
            if test_n_splits == 0:
                raise ValueError('Valid size is too small')

        if groups is not None and verbose:
            print("Received groups value, but randomPerClass don't use it")

        self.n_splits = self.n_repeats

        self.random_state = random_state
        self.iterPos = 1
        self.mask = np.ones(np.asarray(self.y).shape, dtype=bool)

    def __iter__(self):
        return self

    # python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.iterPos < self.n_repeats + 1:

            train, valid = [np.asarray(
                [], dtype=int), np.asarray([], dtype=int)]
            for C in np.unique(self.y):
                Cpos = np.where(self.y == C)[0]
                np.random.seed(self.random_state)

                if self.valid_size < 1:  # means in percent
                    nToKeep = int(self.valid_size * len(Cpos))
                else:
                    nToKeep = self.valid_size

                unMask = np.logical_and(self.y == C, self.mask == 1)
                tmpValid = np.random.permutation(
                    np.where(unMask == 1)[0])[:nToKeep]

                TF = np.in1d(Cpos, tmpValid, invert=True)
                tmpTrain = Cpos[TF]

                train = np.concatenate((train, tmpTrain))
                valid = np.concatenate((valid, tmpValid))

                self.mask[tmpValid] = 0

                unMask = np.logical_and(self.y == C, self.mask == 1)
                if self.valid_size < 1:
                    if np.where(unMask == 1)[0].shape[0] < int(
                            self.valid_size * len(Cpos)):
                        self.mask[np.where(self.y == C)[0]] = 1
                else:
                    if np.where(unMask == 1)[0].shape[0] < self.valid_size:
                        self.mask[np.where(self.y == C)[0]] = 1

            self.random_state += 1
            self.iterPos += 1

            return train, valid
        else:
            raise StopIteration()


class groupCV:
    def __init__(self, X=None, y=None, groups=None, n_repeats=False,
                 valid_size=1, random_state=False, verbose=False):
        """Compute train/validation per group.

        Parameters
        ----------

        X : None
            For sklearn compatiblity
        Y : array-like
            contains class for each ROI.
        groups : array-like
            contains goup number for each ROI (same size of Y).
        valid_size : int (1) or float (0.01 to 0.99)
            If 1 Leave-One-Group Out.
        n_splits : False or int
            if False, n_splits is the minimum stand number of all species.
        SLOO :  Bool
            True  or False. If SLOO, keep only one Y per validation stand.
        """
        self.verbose = verbose
        self.y = y.flatten()
        self.uniqueY = np.unique(self.y)
        self.groups = groups

        self.valid_size = valid_size
        self.iterPos = 1

        self.random_state = random_state

        smallestGroup = []
        for i in np.unique(self.y):
            standNumber = np.unique(
                np.array(groups)[
                    np.where(
                        np.array(self.y).flatten() == i)])

            smallestGroup.append(standNumber.shape[0])
        smallestGroup = np.min(smallestGroup)

        if n_repeats:
            self.n_repeats = n_repeats
        else:
            self.n_repeats = smallestGroup
            if self.n_repeats == 1:
                raise Exception(
                    'You need to have at least two subgroups per label')

        test_n_splits = np.amax(
            (int(valid_size * smallestGroup), int((1 - valid_size) * smallestGroup)))
        if test_n_splits == 0:
            raise ValueError('Valid size is too small')

        self.mask = np.ones(np.asarray(groups).shape, dtype=bool)

        self.n_splits = self.n_repeats

    def __iter__(self):
        return self

    # python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.iterPos < self.n_repeats + 1:
            if self.verbose:
                print(53 * '=')
            train = np.array([], dtype=int)
            validation = np.array([], dtype=int)
            for C in self.uniqueY:
                Ycurrent = np.where(np.array(self.y) == C)[0]
                Ystands = np.array(self.groups)[Ycurrent]

                nYstand = len(np.unique(self.groups[self.y == C]))
                Ystand = self.groups[np.logical_and(
                    self.mask == 1, self.y == C)]
                if self.valid_size >= 1:
                    nToKeep = self.valid_size
                else:
                    nToKeep = nYstand * self.valid_size
                    nToKeep += (nToKeep < 1)
                    nToKeep = int(nToKeep)

                if np.unique(Ystand).shape[0] < nToKeep:
                    # reset mask because not enough group
                    self.mask[Ycurrent] = 1
                    Ystand = self.groups[self.y == C]

                np.random.seed(self.random_state)
                selectedStand = np.random.permutation(
                    np.unique(Ystand))[:nToKeep]

                if self.verbose:
                    print('For class {}, subgroup {}'.format(C, selectedStand))

                YinSelectedStandt = np.in1d(Ystands, selectedStand)
                tmpValid = Ycurrent[YinSelectedStandt]
                validation = np.concatenate(
                    (validation, tmpValid))

                YnotInSelectedStandt = np.invert(YinSelectedStandt)
                tmpTrain = Ycurrent[YnotInSelectedStandt]
                train = np.concatenate(
                    (train, tmpTrain))

                self.mask[tmpValid] = 0
            self.random_state += 1
            self.iterPos += 1
            return train, validation
        else:
            raise StopIteration()


class _cv_manager:
    def __init__(self, cv_type, verbose=False, **params):
        """
        Manage cross-validation methods to generate the duo valid/train samples.

        """
        self.cv_type = cv_type
        self.verbose = verbose

        self.__extensions = ['sqlite', 'shp', 'netcdf', 'gpx', 'gpkg']
        self.__driversName = [
            'SQLITE',
            'ESRI Shapefile',
            'netCDF',
            'GPX',
            'GPKG']

        self.__alertMessage = 'It seems you already generated the cross validation. \n Please use reinitialize function if you want to regenerate the cross validation. \n But check if you defined a random_state number in your samplingMethods function to have the same output.'
        self.__alreadyRead = False

        self.params = params

        # create unique random state if no random_state
        if self.params['random_state'] is False:
            import time
            self.params['random_state'] = int(time.time())

    def reinitialize(self):
        _cv_manager.__init__(
            self, self.cv_type, self.verbose, **self.params)

    def get_supported_extensions(self):
        print('Museo ToolBox supported extensions for writing vector files are : ')
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

        if X is not None:
            X = np.empty(y.reshape(-1, 1).shape, dtype=np.int16)

        if self.cv_type.__name__ == 'distanceCV':
            # TODO : Find a better way to get n_splits for distanceCV
            # As distance may differ from real n_splits, hard to not run the
            # whole thing
            n_splits = 0
            for tr, vl in self.cv_type(
                    X=X, y=y, groups=groups, verbose=self.verbose, **self.params):
                n_splits += 1

        else:
            n_splits = self.cv_type(
                X=X,
                y=y,
                groups=groups,
                verbose=self.verbose,
                **self.params).n_splits

        if n_splits == 0:
            raise ValueError(
                'Sorry but due to your dataset or your distance thresold, no cross-validation has been generated.')

        return n_splits

    def split(self, X, y, groups=None):
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
        y = y.reshape(-1, 1)
        if self.__alreadyRead:
            self.reinitialize()
        self.__alreadyRead = True

        return self.cv_type(
            X=X, y=y, groups=groups, verbose=self.verbose, **self.params)

    def save_to_vector(self, vector, field, group=None, out_vector=None):
        """
        Save to vector files each fold from the cross-validation.

        Parameters
        -------------
        vector : str.
            Path where the vector is stored.
        field : str.
            Name of the field containing the label.
        group : str, or None.
            Name of the field containing the group/subgroup (or None
        out_vector : str.

            Path and filename to save the different results.

        Returns
        ----------
        list_of_output : list
            List containing the number of folds * 2.

            train + validation for each fold.

        """
        src = ogr.Open(vector)
        srcLyr = src.GetLayerByIndex()
        self.wkbType = srcLyr.GetGeomType()
        if self.wkbType != 1:
            print(
                """Warning : This function generates vector files according to your vector.
        The number of features may differ from the number of pixels used in classification.
        If you want to save every ROI pixels in the vector, please use processing.sample_extraction before.""")
        del src, srcLyr

        fileName, self.__ext = os.path.splitext(out_vector)

        if self.__ext[1:] not in self.__extensions:
            print(
                'Your extension {} is not recognized as a valid extension for saving shape.'.format(
                    self.__ext))
            self.get_supported_extensions()
            raise Exception(
                'We recommend you to use gpkg or sqlite extension.')

        if group is None:
            groups = None
            y, fts, srs = processing.read_vector_values(
                vector, field, get_features=True, verbose=self.verbose)
        else:
            y, groups, fts, srs = processing.read_vector_values(
                vector, field, group, get_features=True, verbose=self.verbose)

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
                self.__save_to_vector(trFeat, srs, tr)
                self.__save_to_vector(vlFeat, srs, vl)
                listOutput.append([tr, vl])
        self.__alreadyRead = True
        return listOutput

    def __save_to_vector(self, array, srs, outShapeFile):
        # Parse a delimited text file of volcano data and create a shapefile
        # use a dictionary reader so we can access by field name
        # set up the shapefile driver

        driverIdx = [x for x, i in enumerate(
            self.__extensions) if i == self.__ext[1:]][0]
        outDriver = ogr.GetDriverByName(self.__driversName[driverIdx])

        if os.path.exists(outShapeFile):
            # Remove output shapefile if it already exists
            outDriver.DeleteDataSource(outShapeFile)

        # create the data source
        ds = outDriver.CreateDataSource(outShapeFile)

        # create the spatial reference, WGS84
        lyrout = ds.CreateLayer(
            'cv',
            srs=srs,
            geom_type=self.wkbType)
        fields = [
            array[1].GetFieldDefnRef(i).GetName() for i in range(
                array[1].GetFieldCount())]

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
