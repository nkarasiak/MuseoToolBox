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
"""
The :mod:`museotoolbox.crossValidation` module gathers cross-validation function.
"""
from __future__ import absolute_import, print_function

from ._sampleSelection import _sampleSelection
from . import crossValidationClass as _cvc


class LeavePSubGroupOut(_sampleSelection):
    def __init__(self,
                 valid_size=0.5,
                 n_splits=False,
                 random_state=None,
                 verbose=False):
        """
        Generate a Cross-Validation using subgroup (each group belong to a unique label).

        Parameters
        ----------
        valid_size : float, default 0.5.
            From 0 to 1.
        n_splits : int or False, default False.
            If False, is the number of
        random_state : int or None, default=None.
            If int, random_state is the seed used by the random number generator;
            If None, the random number generator is created with ``time.time()``.
        verbose : int or False, defaultr False.
            Controls the verbosity: the higher, the more messages.
        """
        self.verbose = verbose

        self.crossvalidation = _cvc.groupCV

        if isinstance(valid_size, float):
            if valid_size > 1 or valid_size < 0:
                raise Exception('Percent must be between 0 and 1')
        else:
            raise Exception(
                'Percent must be between 0 and 1 and must be a float')

        self.params = dict(
            valid_size=valid_size,
            n_splits=n_splits,
            random_state=random_state)

        _sampleSelection.__init__(self)


class LeaveOneSubGroupOut(_sampleSelection):
    def __init__(self,
                 n_splits=False,
                 random_state=None,
                 verbose=False):
        """
        Generate a Cross-Validation by subgroup.

        Parameters
        ----------
        inVector : str or array.
            if str, path of vector file, if array same size as inGroup array.
        inField : str or None.
            if str, will extract value from inVector file.
        inGroup : str or array.
            field name containing label of groups. If array, same size as inVector array.
        n_splits : default False.
            If False : will iterate as many times as the smallest number of groups.
            If int : will iterate the number of times given in n_splits.
        random_state : int, default None.
            If random_state, int, to repeat exactly the same random.


        Returns
        -------
        List : list with the sampling type and the parameters for the groupCV.
        """
        self.verbose = verbose

        self.crossvalidation = _cvc.groupCV

        self.params = dict(
            valid_size=1,
            n_splits=n_splits,
            random_state=random_state)
        _sampleSelection.__init__(self)


class SpatialLeavePSideOut(_sampleSelection):
    def __init__(self,
                 distanceMatrix=None,
                 minTrain=None,
                 n_splits=False,
                 random_state=None,
                 verbose=False):
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
            If False : will iterate as many times as the smallest number of groups.
            If int : will iterate the number of groups given in maxIter.
        random_state : int, default None.
            If random_state, int, to repeat exactly the same random.

        Returns
        -------
        List : list with the sampling type and the parameters for the farthestCV.
        """
        self.samplingType = 'Spatial'
        self.verbose = verbose

        self.crossvalidation = _cvc.distanceCV

        self.params = dict(
            distanceMatrix=distanceMatrix,
            minTrain=minTrain,
            n_splits=n_splits,
            random_state=random_state,
            furtherSplit=True)
        _sampleSelection.__init__(self)


class SpatialLeaveOneSubGroupOut(_sampleSelection):
    def __init__(self,
                 distanceThresold=None,
                 distanceMatrix=None,
                 distanceLabel=None,
                 n_splits=False,
                 random_state=None,
                 verbose=False):
        """
        Generate a Cross-Validation with Spatial Leave-One-Out method.

        Parameters
        ----------
        distanceMatrix : None or array.
            If array got from function vectorTools.getDistanceMatrix(inRaster,inVector)
        distanceThresold : int.
            In pixels.
        minTrain : int/float, default None.
            The minimum of training pixel to achieve. if float (0.01 to 0.99) will a percentange of the training pixels.
        SLOO : True or float
            from 0.0 to 1.0 (means keep 90% for training). If True, keep only one sample per class for validation.
        n_splits : default False.
            If False : will iterate as many times as the smallest number of groups.
            If int : will iterate the number of groups given in maxIter.
        random_state : int, default None.
            If random_state, int, to repeat exactly the same random.

        Returns
        -------
        List : list with the sampling type and the parameters for the SLOOCV.

        References
        ----------
        See : https://doi.org/10.1111/geb.12161.
        """
        self.samplingType = 'Spatial'
        self.verbose = verbose

        self.crossvalidation = _cvc.distanceCV

        self.params = dict(
            distanceMatrix=distanceMatrix,
            distanceThresold=distanceThresold,
            distanceLabel=distanceLabel,
            SLOO=True,
            n_splits=n_splits,
            random_state=random_state)
        _sampleSelection.__init__(self)


class SpatialLeaveOnePixelOut(_sampleSelection):
    """
    Generate a Cross-Validation with Spatial Leave-One-Out method.

    Parameters
    ----------
    distanceMatrix : array.
        Array got from function samplingMethods.getDistanceMatrixForDistanceCV(inRaster,inVector)
    distanceThresold : int.
        In pixels.
    minTrain : int/float, default None.
        The minimum of training pixel to achieve. if float (0.01 to 0.99) will a percentange of the training pixels.
    SLOO : True or float
        from 0.0 to 1.0 (means keep 90% for training). If True, keep only one sample per class for validation.
    n_splits : default False.
        If False : will iterate as many times as the smallest number of groups.
        If int : will iterate the number of groups given in maxIter.
    random_state : int, default None.
        If random_state, int, to repeat exactly the same random.

    Returns
    --------
    List : list with the sampling type and the parameters for the SLOOCV.

    References
    ----------
    See : https://doi.org/10.1111/geb.12161.
    """

    def __init__(self,
                 distanceThresold=None,
                 distanceMatrix=None,
                 distanceLabel=False,
                 n_splits=False,
                 random_state=None,
                 verbose=False):

        self.samplingType = 'Spatial'
        self.verbose = verbose

        self.crossvalidation = _cvc.distanceCV

        self.params = dict(
            distanceMatrix=distanceMatrix,
            distanceThresold=distanceThresold,
            distanceLabel=distanceLabel,
            minTrain=False,
            SLOO=True,
            n_splits=n_splits,
            random_state=random_state)
        _sampleSelection.__init__(self)


class RandomCV(_sampleSelection):
    def __init__(self,
                 valid_size=0.5,
                 n_splits=False,
                 random_state=None,
                 verbose=False):
        """
        Get parameters to have a randomCV.

        Parameters
        ----------

        split : float,int. Default 0.5.
            If float from 0.1 to 0.9 (means keep 90% per class for training). If int, will try to reach this sample for every class.
        nSamples: int or str. Default None.
            If int, the max samples per class.
            If str, only 'smallest' to sample as the smallest class.
        random_state : int, default None.
            If random_state, int, to repeat exactly the same random.
        n_splits : int, default 5.
            Number of iteration of the random sampling (will add 1 to the random_state at each iteration if defined).

        Returns
        --------
        List : list with the sampling type and the parameters for the randomCV.

        """
        self.samplingType = 'random'
        self.verbose = verbose

        self.crossvalidation = _cvc.randomPerClass

        self.params = dict(
            valid_size=valid_size,
            random_state=random_state,
            n_splits=n_splits)

        _sampleSelection.__init__(self)
