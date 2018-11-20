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
import numpy as np

class distanceCV:
    def __init__(
            self,
            distanceMatrix,
            Y,
            distanceThresold=1000,
            minTrain=-1,
            SLOO=True,
            n_splits=False,
            furtherSplit=False,
            onlyVaryingTrain=False,
            stats=False,
            verbose=False,
            seed=False,
            otherLevel=False,
            distanceMatrixLabel=False):
        """Compute train/validation array with Spatial distance analysis.

        Object stops when less effective class number is reached (45 loops if your least class contains 45 ROI).

        Parameters
        ----------
        distanceMatrix : array
            Matrix distance
        Y : array-like
            contain class for each ROI. Same effective as distanceArray.
        distanceThresold : int or float
            Distance(same unit of your distanceArray)
        minTrain : int or float
            if >1 : keep n ROI beyond distanceThresold
            if float <1 : minimum percent of samples to use for traning. Use 1 for use only distance Thresold.
            if -1 : same size
        SLOO : Spatial Leave One Out, keep on single validation pixel.
            SLOO=True: Skcv (if n_splits=False, skcv is SLOO from Kevin Le Rest, or SKCV from Pohjankukka)
        n_splits :
            False : as loop as min effective class
        otherLevel : array
            contain class (like Y), e.g. able to do a SLOO per Stand if you put your stand number here.

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
        self.label = np.copy(Y)
        self.T = np.copy(Y)
        self.iterPos = 0
        self.minTrain = minTrain
        self.distanceMatrixLabel = distanceMatrixLabel
        if self.minTrain is None:
            self.minTrain = -1
        self.onlyVaryingTrain = onlyVaryingTrain
        if self.onlyVaryingTrain:
            self.validation = np.array([]).astype('int')
        self.otherLevel = otherLevel
        if self.otherLevel is not False and self.distanceMatrixLabel is False:
            raise Exception('You need the to set the distanceMatrixLabel in order to compute spatial leave-one-out method using a subclass.')
        self.SLOO = SLOO  # Leave One OUT
        self.verbose = verbose
        self.furtherSplit = furtherSplit
        self.stats = stats
        if self.otherLevel is False:
            self.minEffectiveClass = min([len(Y[Y == i]) for i in np.unique(Y)])
        else:    
            self.minEffectiveClass = min([len(np.unique(otherLevel[np.where(Y==i)[0]])) for i in np.unique(Y)])
        if n_splits:
            self.n_splits = n_splits
        else:
            self.n_splits = self.minEffectiveClass

        if seed:
            np.random.seed(seed)
        else:
            import time
            seed = int(time.time())
        self.seed = seed
        self.mask = np.array([])

    def __iter__(self):
        return self

    # python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        #global CTtoRemove,trained,validate,validation,train,CT,distanceROI
        if self.iterPos < self.n_splits:
            np.random.seed(self.seed)
            self.seed += 1
            ROItoRemove = []
            for iterPosition in range(self.n_splits):
                if self.verbose:
                    print((53 * '=' + '\n') * 4)

                validation = np.array([]).astype('int')
                train = np.array([]).astype('int')

                if self.stats:
                    #Cstats = sp.array([]).reshape(0,9   )
                    Cstats = []

                for C in np.unique(self.label):

                    # Y is True, where C is the unique class
                    allCT = np.where(self.label == C)[0]
                    
                    CT = np.where(self.T == C)[0]

                    CTtemp = np.copy(CT)

                    
                    if self.verbose:
                        print('C = ' + str(C))
                        print('len total class : ' + str(len(CT)))
                        fmt = '' if self.minTrain > 1 else '.0%'

                    trained = np.array([]).astype('int')
                    validate = np.array([]).astype('int')
                    
                    while len(CTtemp) > 0:
                        
                        self.ROI = np.random.permutation(CTtemp)[0]
                        if self.otherLevel is not False:
                            while self.otherLevel[self.ROI] in self.mask:
                                self.ROI = np.random.permutation(CTtemp)[0]
                        else:
                            while self.ROI in self.mask:
                                self.ROI = np.random.permutation(CTtemp)[0]

                        if self.otherLevel is not False and self.distanceMatrixLabel is not None:
                            Tstand =  self.distanceMatrixLabel[np.isin(self.distanceMatrixLabel,np.unique(self.otherLevel[allCT]))]
                            TstandTF = np.isin(self.distanceMatrixLabel,Tstand)
                            standPos = np.argwhere(self.otherLevel[self.ROI]==self.distanceMatrixLabel)[0][0]
                            distanceROI = (self.distanceArray[standPos,:])
                            distanceROI = distanceROI[TstandTF]
                            
                        else:    
                            distanceROI = (self.distanceArray[int(self.ROI), :])[allCT]  # get line of distance for specific ROI

                        if self.minTrain == -1:
                            if self.otherLevel is not False:
                                trainedTemp = np.where(self.otherLevel == self.otherLevel[self.ROI])[0]
                                #validateTStand = distanceROI[np.where(distanceROI>= self.distanceThresold)[0]]
                                validateTemp = allCT[np.isin(self.otherLevel[allCT],self.distanceMatrixLabel[np.where(distanceROI>= self.distanceThresold)[0]])]

                            else:
                                trainedTemp = np.array([self.ROI])

                                validateTemp = allCT[distanceROI >
                                                  self.distanceThresold]
                            # CTtemp[distanceROI>=self.distanceThresold] #
                            # trained > distance to cut

                        if self.n_splits == self.minEffectiveClass:
                            if self.otherLevel is not False:
                                trainedTemp = np.where(self.otherLevel == self.otherLevel[self.ROI])[0]
                                #validateTStand = distanceROI[np.where(distanceROI>= self.distanceThresold)[0]]
                                validateTemp = allCT[np.isin(self.otherLevel[allCT],self.distanceMatrixLabel[np.where(distanceROI>= self.distanceThresold)[0]])]

                            else:
                                trainedTemp = np.array([self.ROI])

                                validateTemp = CTtemp[distanceROI >
                                                  self.distanceThresold]
                            #trainedTemp = trainedTemp[trainedTemp!=self.ROI]

                        if self.furtherSplit:
                            validateTemp = np.array([self.ROI])
                            trainedTemp = CTtemp[CTtemp != validateTemp]

                        trained = trainedTemp
                        validate = validateTemp

                        CTtemp = []
                        #print len(validate)
                    initTrain = len(trained)
                    initValid = len(validate)

                    if self.minTrain > 1:
                        if len(trained) != self.minTrain:

                            # get number of ROI to keep
                            indToCut = len(CT) - int(self.minTrain)
                            # get distance where to split train/valid
                            distToCut = np.sort(distanceROI)[indToCut]
                            # trained > distance to cut
                            trained = CT[distanceROI >= distToCut]

                            if self.SLOO:  # with SLOO we keep 1 single validation ROI
                                trained = np.random.permutation(
                                    trained)[0:self.minTrain]
                            else:
                                if self.verbose:
                                    print('len validate before split (' +
                                          format(self.minTrain, fmt) +
                                          ') : ' +
                                          str(len(validate)))
                                validate = CT[distanceROI <= distToCut]

                    if self.stats:

                        #CTtemp = sp.where(self.label[trained]==C)[0]
                        CTdistTrain = np.array(self.distanceArray[trained])[
                            :, trained]
                        if len(CTdistTrain) > 1:
                            CTdistTrain = np.mean(
                                np.triu(CTdistTrain)[
                                    np.triu(CTdistTrain) != 0])

                        #CTtemp = sp.where(self.label[validate]==C)[0]
                        CTdistValid = np.array(self.distanceArray[validate])[
                            :, validate]
                        CTdistValid = np.mean(
                            np.triu(CTdistValid)[
                                np.triu(CTdistValid) != 0])
                        Cstats.append([self.distanceThresold,
                                       self.minTrain,
                                       C,
                                       initValid,
                                       initTrain,
                                       len(trained) - initTrain,
                                       CTdistTrain,
                                       CTdistValid])

                    if self.verbose:
                        print('len validate : ' + str(len(validate)))
                        print('len trained : ' + str(len(trained)))

                    validation = np.concatenate((validation, validate))
                    train = np.concatenate((train, trained))

                    # remove current validation ROI
                    #ROItoRemove.append(validation)
                    ROItoRemove.append(train)

                if self.stats is True:
                    np.savetxt(
                        self.stats,
                        Cstats,
                        fmt='%d',
                        delimiter=',',
                        header="Distance,Percent Train, Label,Init train,Init valid,Ntrain Add,Mean DisT Train,Mean Dist Valid")

                self.iterPos += 1

                if self.verbose:
                    print(53 * '=')
                
                # Remove ROI for further selection ROI (but keep in Y list)
                if self.otherLevel is not False:
                    self.mask = np.concatenate((self.mask,np.unique(self.otherLevel[train])))
                else:
                    self.mask = np.concatenate((self.mask,np.unique(train)))

                
                if train.shape[0]>0:
                    if self.stats and self.stats is True:
                        return validation, train, Cstats
                    else:
                        return validation, train

        else:
            raise StopIteration()


class randomPerClass:
    """
    Random array according to FIDs.

    Parameters
    ----------
    FIDs : arr.
        Label for each feature.
    train_size : float (<1.0) or int (>1).
        Percentage to keep for training or integer.
    valid_size : False or int
        1 to do a Leave-One-Out.
    seed : int.
        random_state for numpy.
        
    """

    def __init__(self, FIDs, train_size=0.5,valid_size=False, n_splits=5, seed=None):
        self.name = 'randomPerClass'
        self.FIDs = FIDs
        self.train_size = train_size
        self.n_splits = n_splits
        if n_splits is False:
            n_splits = min([len(self.FIDs==C) for C in np.unique(FIDs)])
        if seed:
            np.random.seed(seed)
        else:
            print('No seed defined, will use time')
            import time
            seed = int(time.time())
        self.seed = seed
        self.valid_size= valid_size
        self.iter = 0
        self.mask = np.ones(np.asarray(self.FIDs).shape,dtype=bool)

    def __iter__(self):
        return self

    # python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.iter < self.n_splits:
            np.random.seed(self.seed)
            train, valid = [np.asarray([],dtype=int), np.asarray([],dtype=int)]
            for C in np.unique(self.FIDs):
                Cpos = np.where(self.FIDs == C)[0]
                if self.train_size < 1:
                    toSplit = int(self.train_size * len(Cpos))
                else:
                    toSplit = self.train_size

                if self.valid_size>=1:
                    tempValid = np.asarray([np.random.permutation(Cpos)[:self.valid_size]]).flatten()
                    TF = np.in1d(Cpos, tempValid, invert=True)
                    tempTrain = Cpos[TF]
                    
                else:
                    tempTrain = np.random.permutation(Cpos)[:toSplit]
                    TF = np.in1d(Cpos, tempTrain, invert=True)
                    tempValid = Cpos[TF]
                train = np.concatenate((train, tempTrain))
                valid = np.concatenate((valid, tempValid))
                
                self.mask[valid]=False

            self.seed += 1
            self.iter += 1

            return train, valid
        else:
            raise StopIteration()

class standCV:
    def __init__(self, Y, stand, n_splits=False, valid_size=1, seed=False):
        """Compute train/validation per stand.
        Y : array-like
            contains class for each ROI.
        Stand : array-like
            contains stand number for each ROI.
        valid_size : int (1) or float (0.01 to 0.99)
            If 1 Leave-One-Group Out.
        n_splits : False or int
            if False, n_splits is the minimum stand number of all species.
        SLOO :  Bool
            True  or False. If SLOO, keep only one Y per validation stand.
        """
        self.name = 'standCV'
        self.Y = Y
        self.uniqueY = np.unique(self.Y)
        self.stand = stand

        self.valid_size = valid_size
        self.n_splits = n_splits
        self.iterPos = 0

        if seed:
            np.random.seed(seed)
        else:
            print('No seed defined, will use time')
            import time
            seed = int(time.time())
        self.seed = seed
        if n_splits:
            self.n_splits = n_splits
        else:
            n_splits = []
            for i in np.unique(Y):
                standNumber = np.unique(
                    np.array(stand)[
                        np.where(
                            np.array(Y) == i)])
                n_splits.append(standNumber.shape[0])
            self.n_splits = np.amin(n_splits)
            if self.n_splits == 1:
                raise Exception('You need to have at least two subgroups per label')
        
        self.mask = np.ones(np.asarray(stand).shape,dtype=bool)

    def __iter__(self):
        return self

    # python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.iterPos < self.n_splits:
            train = np.array([], dtype=int)
            validation = np.array([], dtype=int)
            for i in self.uniqueY:
                Ycurrent = np.where(np.array(self.Y) == i)[0]
                Ystands = np.array(self.stand)[Ycurrent]

                np.random.seed(self.seed)
                #Only choose an unselected stand
                YTF = np.array(self.Y)==i
                Ystand = np.unique(Ystands)

                if len(Ystands[self.mask[Ycurrent]])==0:
                    #Reset mask if not enought subgroup available
                    #Appear only if n_splits > min len of subgroup
                    self.mask[:] = 1
                    
                if self.valid_size ==1:
                    selectedStand = np.random.permutation(Ystands[self.mask[Ycurrent]])[0]
                    
                if self.valid_size <1:
                    selectedStand = np.random.permutation(Ystands)[:int(len(Ystand)*self.valid_size)]
                    
                YinSelectedStandt = np.in1d(Ystands, selectedStand)
                YinSelectedStand = Ycurrent[YinSelectedStandt]
                validation = np.concatenate(
                    (validation, np.asarray(YinSelectedStand)))

                YnotInSelectedStandt = np.invert(YinSelectedStandt)
                YnotInSelectedStand = Ycurrent[YnotInSelectedStandt]
                train = np.concatenate(
                    (train, np.asarray(YnotInSelectedStand)))
                
                if self.valid_size==1:
                    self.mask = np.logical_and(self.mask,~np.logical_and(YTF,np.in1d(self.stand,selectedStand)))
                
                self.seed += 1
            self.iterPos += 1
            return train, validation
        else:
            raise StopIteration()


