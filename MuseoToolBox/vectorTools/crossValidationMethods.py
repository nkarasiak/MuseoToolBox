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
from ..vectorTools.crossValidationSelection import sampleSelection
from ..vectorTools import getDistanceMatrixForDistanceCV

class LeavePercentSubStandOut(sampleSelection):
    def __init__(self,
             inVector,
             inStand,
             inField=None,
             percent=0.5,
             maxIter=5, 
             seed=None):
        """
        Generate a Cross-Validation by stand/polygon group.

        Parameters
        ----------
        inStand : str.
            field name containing label of group/stand.
        SLOO : float, default True.
            If float from 0.1 to 0.9 (means keep 90% for training)
        maxIter : default False.
            If False : will iterate as many times as the smallest number of stands.
            If int : will iterate the number of stands given in maxIter.

        seed : int, default None.
            If seed, int, to repeat exactly the same random.
            
        Returns
        -------
        List : list with the sampling type and the parameters for the standCV.
        """
        self.samplingType = 'STAND'
        if isinstance(SLOO,float):
            if percent>1 or percent<1:
                raise Exception('Percent must be between 0 and 1')
        else:
            raise Exception('Percent must be between 0 and 1 and must be a float')
            
        self.params = dict(
                inVector=inVector,
                inField=inField,
                inStand=inStand,
                SLOO=percent,
                maxIter=maxIter,
                seed=seed)
        sampleSelection.__init__(self)

class LeaveOneSubStandOut(sampleSelection):
    def __init__(self,
             inVector,
             inStand,
             inField=None,
             SLOO=True,
             maxIter=False, 
             seed=None):
        """
        Generate a Cross-Validation by stand/polygon group.

        Parameters
        ----------
        inStand : str.
            field name containing label of group/stand.
        SLOO : float, default True.
            If float from 0.1 to 0.9 (means keep 90% for training)
        maxIter : default False.
            If False : will iterate as many times as the smallest number of stands.
            If int : will iterate the number of stands given in maxIter.

        seed : int, default None.
            If seed, int, to repeat exactly the same random.
            
        Returns
        -------
        List : list with the sampling type and the parameters for the standCV.
        """
        self.samplingType = 'STAND'
        self.params = dict(
                inStand=inStand,
                SLOO=SLOO,
                maxIter=maxIter,
                seed=seed)
        sampleSelection.__init__(self)

class farthestCV(sampleSelection):
    def __init__(self,
            inRaster,
            inVector,
            distanceMatrix,
            inField=None,
            minTrain=None,
            maxIter=False,
            seed=None):
        """
        Generate a Cross-Validation using the farthest distance between the training and validation samples.

        Parameters
        ----------
        inRaster : str.
            Path of the raster.
        inVector : str.
            Path of the vector.
        distanceMatrix : array.
            Array got from function samplingMethods.getDistanceMatrixForDistanceCV(inRaster,inVector)
        minTrain : int/float, default None.
            The minimum of training pixel to achieve. if float (0.01 to 0.99) will a percentange of the training pixels.
        maxIter : default False.
            If False : will iterate as many times as the smallest number of stands.
            If int : will iterate the number of stands given in maxIter.
        seed : int, default None.
            If seed, int, to repeat exactly the same random.
            
        Returns
        -------
        List : list with the sampling type and the parameters for the farthestCV.
        """

        self.samplingType = 'farthestCV'
        self.params = dict(
                inRaster=inRaster,
                inVector=inVector,
                inField=inField,
                distanceMatrix=distanceMatrix,
                minTrain=minTrain,
                maxIter=maxIter,
                seed=seed,
                furtherSplit=True)
        sampleSelection.__init__(self)

class SLOO(sampleSelection):
    def __init__(self,
            inRaster,
            inVector,
            distanceThresold,
            inField=None,
            distanceMatrix=None,
            minTrain=None,
            SLOO=True,
            n_splits=False,
            seed=None):
        """
        Generate a Cross-Validation with Spatial Leave-One-Out method.

        Parameters
        ----------
        inRaster : str.
            Path of the raster.
        inVector : str.
            Path of the vector.
        distanceMatrix : array.
            Array got from function samplingMethods.getDistanceMatrixForDistanceCV(inRaster,inVector)
        distanceThresold : int.
            In pixels.
        minTrain : int/float, default None.
            The minimum of training pixel to achieve. if float (0.01 to 0.99) will a percentange of the training pixels.
        SLOO : True or float
            from 0.0 to 1.0 (means keep 90% for training). If True, keep only one sample per class for validation.
        n_splits : default False.
            If False : will iterate as many times as the smallest number of stands.
            If int : will iterate the number of stands given in maxIter.
        seed : int, default None.
            If seed, int, to repeat exactly the same random.
            
        Returns
        -------
        List : list with the sampling type and the parameters for the SLOOCV.
        
        References
        ----------
        See : https://doi.org/10.1111/geb.12161.
        """
        if distanceMatrix is None:
            distanceMatrix = getDistanceMatrixForDistanceCV(
                inRaster, inVector)

        self.samplingType = 'SLOO'
        self.params = dict(
                inRaster=inRaster,
                inVector=inVector,
                inField=inField,
                distanceMatrix=distanceMatrix,
                distanceThresold=distanceThresold,
                minTrain=minTrain,
                SLOO=SLOO,
                n_splits=n_splits,
                seed=seed)
        sampleSelection.__init__(self)

class randomCV(sampleSelection):
    def __init__(self,
            inVector,
            inField=None,
            train_size=0.5,
            n_splits=5,
            seed=None):
        """
        Get parameters to have a randomCV.
        
        Parameters
        ----------
        
        split : float,int. Default 0.5.
            If float from 0.1 to 0.9 (means keep 90% per class for training). If int, will try to reach this sample for every class.
        nSamples: int or str. Default None.
            If int, the max samples per class.
            If str, only 'smallest' to sample as the smallest class.
        seed : int, default None.
            If seed, int, to repeat exactly the same random.
        nIter : int, default 5.
            Number of iteration of the random sampling (will add 1 to the seed at each iteration if defined).
            
        Returns
        --------
        List : list with the sampling type and the parameters for the randomCV.
        
        """
        self.samplingType = 'random'
        self.inVector = inVector
        
        self.params = dict(
                train_size=train_size,
                seed=seed,
                n_splits=n_splits)
        
        sampleSelection.__init__(self)