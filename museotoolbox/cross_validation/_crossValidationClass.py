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
import numpy as np

from itertools import tee


class distanceCV:
    def __init__(
            self,
            X=None,
            y=None,
            distanceMatrix=None,
            distanceThresold=1000,
            minTrain=False,
            SLOO=True,
            n_splits=False,
            valid_size=False,
            stats=False,
            verbose=False,
            random_state=False,
            groups=None,
            distanceLabel=False,
            LOOsameSize=False):
        """Compute train/validation array with Spatial distance analysis.

        Object stops when less effective class number is reached (45 loops if your least class contains 45 ROI).

        Parameters
        ----------
        X : array-like
            Matrix of values. As many row as the Y array.
        Y : array-like
            contain class for each ROI. Same effective as distanceArray.
        distanceMatrix : array
            Matrix distance
        distanceThresold : int or float
            Distance(same unit of your distanceArray)
        minTrain : int or float
            if >1 : keep n ROI beyond distanceThresold
            if float <1 : minimum percent of samples to use for traning. Use 1 for use only distance Thresold.
            if False keep same size.
        SLOO : Spatial Leave One Out, keep on single validation pixel.
            SLOO=True: Skcv (if n_splits=False, skcv is SLOO from Kevin Le Rest, or SKCV from Pohjankukka)
        n_splits :
            False : as loop as min effective class
        valid_size : False or float value.
            If float, value from 0 to 1 (percent).
        groups : array
            contain class (like Y), e.g. able to do a SLOO per Stand if you put your stand number here.
        stat : str.
            Path where to save csv stat (label, n trains, and mean distance between training per class).

        Returns
        -------
        train : array
            List of Y selected ROI for train

        validation : array
            List of Y selected ROI for validation

        """
        self.name = 'SLOO'

        self.distanceArray = distanceMatrix
        self.distanceThresold = distanceThresold
        self.y = y
        self.iterPos = 0
        self.minTrain = minTrain
        self.distanceLabel = distanceLabel
        if self.minTrain is None:
            self.minTrain = -1
        self.valid_size = valid_size
        self.LOOsameSize = LOOsameSize
        self.groups = groups
        if self.groups is not None and self.distanceLabel is False:
            raise Exception(
                'You need the to set the distanceLabel in order to compute spatial leave-one-out method using a subclass.')
        self.SLOO = SLOO  # Leave One OUT
        self.verbose = verbose
        self.stats = stats

        self.nTries = 0

        self.random_state = random_state

        self.mask = np.ones(np.asarray(self.y).shape, dtype=bool)

        if self.stats:
            self.Cstats = []

        if self.groups is None:
            self.minEffectiveClass = min(
                [len(self.y[self.y == i]) for i in np.unique(self.y)])
        else:
            self.minEffectiveClass = min(
                [len(np.unique(groups[np.where(y == i)[0]])) for i in np.unique(self.y)])

        if n_splits:
            self.n_splits = n_splits
            tee
            if self.n_splits > self.minEffectiveClass:
                if self.verbose:
                    print(
                        'Warning : n_splits is superior to the number of unique samples/groups')
        else:
            # TODO : run self.__next__() to get real n_splits as it depends on distance
            # but if running self.__next__() here, iterator will be empty
            # after.
            self.n_splits = self.minEffectiveClass
        if self.verbose:
            print('n_splits:' + str(self.n_splits))

    def __iter__(self):
        return self

    # python 3 compatibility

    def __next__(self):
        return self.next()

    def next(self):
        emptyTrain = False

        if self.iterPos < self.n_splits:
            self.nTries = 0
            completeTrain = False
            while completeTrain is False:
                if self.nTries < 100:
                    emptyTrain = False
                    if self.verbose:
                        print(53 * '=')
                    validation, train = np.array([[], []], dtype=np.int64)
                    for C in np.unique(self.y):
                        # Y is True, where C is the unique class
                        CT = np.where(self.y == C)[0]

                        currentCT = np.logical_and(self.y == C, self.mask == 1)

                        if np.where(currentCT)[
                                0].shape[0] == 0:  # means no more ROI
                            if self.minEffectiveClass == self.n_splits:
                                raise StopIteration()
                            if self.verbose > 1:
                                print(
                                    str(C) + ' has no more valid pixel.\nResetting label mask.')
                            self.mask[self.y == C] = 1
                            currentCT = np.logical_and(
                                self.y == C, self.mask == 1)

                        if np.where(currentCT)[
                                0].shape[0] == 0:  # means no more ROI
                            if self.minEffectiveClass == self.n_splits:
                                raise StopIteration()
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

                            np.random.seed(self.random_state)
                            self.ROI = np.random.permutation(CT)[0]
                            # Tstand = self.distanceLabel[np.isin(
                            #   self.distanceLabel, np.unique(self.groups[CT]))]
                            #TstandTF = np.isin(self.distanceLabel, Tstand)
                            standPos = np.argwhere(
                                self.groups[self.ROI] == self.distanceLabel)[0][0]
                            distanceROI = (self.distanceArray[standPos, :])
                            #distanceROI = distanceROI[TstandTF]
                            tmpValid = np.where(
                                self.groups == self.groups[self.ROI])[0].astype(np.int64)
                            #validateTStand = distanceROI[np.where(distanceROI>= self.distanceThresold)[0]]
                            tmpTrainGroup = np.unique(
                                self.distanceLabel[np.where(distanceROI >= self.distanceThresold)[0]])
                            tmpTrainGroup = tmpTrainGroup[np.isin(
                                tmpTrainGroup, self.groups[CT])]
                            tmpTrain = np.where(np.in1d(self.groups, tmpTrainGroup))[
                                0].flatten()

                            if tmpTrain.shape[0] == 0:
                                emptyTrain = True
                        # When doing Leave-One-Out per pixel
                        else:

                            distanceROI = (self.distanceArray[self.ROI, :])[
                                CT]  # get line of distance for specific ROI
                            if self.valid_size is False:
                                tmpValid = np.array(
                                    [self.ROI], dtype=np.int64)
                                tmpTrain = CT[distanceROI >
                                              self.distanceThresold]
                                if self.LOOsameSize is True:
                                    np.random.seed(self.random_state)
                                    tmpTrain = np.random.permutation(
                                        CT)[:len(CT[distanceROI > self.distanceThresold])]

                            else:
                                if self.valid_size >= 1:
                                    nToCut = self.valid_size
                                else:
                                    nToCut = int(self.valid_size * len(CT))

                                distanceToCut = np.sort(distanceROI)[
                                    :nToCut][-1]

                                tmpValid = CT[distanceROI <=
                                              distanceToCut]
                                tmpTrain = CT[distanceROI >
                                              distanceToCut]

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
                            if not np.all(self.y[tmpTrain]) or self.y[tmpTrain][0] != C or not np.all(
                                    self.y[tmpValid]) or self.y[tmpValid][0] != C:
                                raise IndexError(
                                    'Selected labels do not correspond to selected class, please leave feedback')
                            #
                            validation = np.concatenate((validation, tmpValid))
                            train = np.concatenate((train, tmpTrain))
                            if self.stats:
                                CTdistTrain = np.array(self.distanceArray[tmpTrain])[
                                    :, tmpTrain]
                                if len(CTdistTrain) > 1:
                                    CTdistTrain = np.mean(np.triu(CTdistTrain)[
                                                          np.triu(CTdistTrain) != 0])
                                self.Cstats.append(
                                    [C, tmpTrain.shape[0], CTdistTrain])

                    if self.verbose:
                        print('Validation samples : ' + str(len(validation)))
                        print('Training samples : ' + str(len(train)))
                    if self.stats:
                        np.savetxt(self.stats, self.Cstats, fmt='%d',
                                   header="Label,Ntrain,Mean dist train")

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
                    raise AttributeError(
                        'Error : Not enough samples using this distance/valid_size.')
                    raise StopIteration()
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
                 valid_size=0.5, train_size=None, n_splits=False,
                 random_state=None, verbose=False):

        self.name = 'randomPerClass'
        self.y = y
        self.valid_size = valid_size
        self.train_size = 1 - self.valid_size

        smallestClass = np.min(np.unique(y, return_counts=True)[1])

        if n_splits is False:
            if self.valid_size >= 1:
                self.n_splits = smallestClass
            else:
                self.n_splits = int(1 / self.valid_size)
        else:
            self.n_splits = n_splits

        if self.valid_size < 1:
            test_n_splits = int(valid_size * smallestClass)
            if test_n_splits == 0:
                raise ValueError('Valid size is too small')

        if groups is not None:
            print("Received groups value, but randomCV don't use it")

        self.random_state = random_state
        self.iterPos = 1
        self.mask = np.ones(np.asarray(self.y).shape, dtype=bool)

    def __iter__(self):
        return self

    # python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.iterPos < self.n_splits + 1:

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

                if not np.all(self.y[tmpTrain] + 1) or self.y[tmpTrain][0] != C or not np.all(
                        self.y[tmpValid] + 1) or self.y[tmpValid][0] != C:
                    raise IndexError(
                        'Selected labels do not correspond to selected class, please leave feedback')

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
    def __init__(self, X=None, y=None, groups=None, n_splits=False,
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
        self.name = 'standCV'
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

        if n_splits:
            self.n_splits = n_splits
        else:
            self.n_splits = smallestGroup
            if self.n_splits == 1:
                raise Exception(
                    'You need to have at least two subgroups per label')

        test_n_splits = np.amax(
            (int(valid_size * smallestGroup), int((1 - valid_size) * smallestGroup)))
        if test_n_splits == 0:
            raise ValueError('Valid size is too small')

        self.mask = np.ones(np.asarray(groups).shape, dtype=bool)

    def __iter__(self):
        return self

    # python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.iterPos < self.n_splits + 1:
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

                if not np.all(self.y[tmpTrain] + 1) or self.y[tmpTrain][0] != C or not np.all(
                        self.y[tmpValid] + 1) or self.y[tmpValid][0] != C:
                    raise IndexError(
                        'Selected labels do not correspond to selected class, please leave feedback')
                self.mask[tmpValid] = 0
            self.random_state += 1
            self.iterPos += 1
            return train, validation
        else:
            raise StopIteration()
