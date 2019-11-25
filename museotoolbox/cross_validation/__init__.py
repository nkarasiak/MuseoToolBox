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
The :mod:`museotoolbox.cross_validation` module gathers cross-validation classes.
"""
from ._sampleSelection import _sampleSelection
from . import _crossValidationClass as _cvc


class LeaveOneOut(_sampleSelection):
    """
    Generate a Cross-Validation using a Stratified Leave One Out.
    Note : :class:`~LeaveOneOut` is equivalent to :class:``~RandomCV(valid_size=1)``
    
    :class:`~RandomCV(valid_size=1)`
    :class:`~LeavePSubGroupOut`
    `LeavePSubGroupOut:valid_size=1`
    ``~LeavePSubGroupOut(valid_size=1)``
    ``~LeavePSubGroupOut(valid_size=1)``
    :mod:``~LeavePSubGroupOut(valid_size=1)``
    :mod:``LeavePSubGroupOut(valid_size=1)``
    :mod:`LeavePSubGroupOut(valid_size=1)`
    :mod:`~LeavePSubGroupOut(valid_size=1)`
    
    
    Parameters
    ----------
    n_splits : int or bool, optional (default=False).
        If False : will iterate as many times as the smallest class.
        If int : will iterate the number of times given in n_splits.
    random_state : integer or None, optional (default=None).
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is created with ``time.time()``.
    verbose : integer or False, optional (default=False).
        Controls the verbosity: the higher the value is, the more the messages are detailed.
    """

    def __init__(self,
                 n_splits=False,
                 random_state=None,
                 verbose=False):
        self.verbose = verbose

        self.crossvalidation = _cvc.randomPerClass

        self.params = dict(
            valid_size=1,
            n_splits=n_splits,
            random_state=random_state)

        _sampleSelection.__init__(self)


class LeavePSubGroupOut(_sampleSelection):
    """
    Generate a Cross-Validation using subgroup (each group belong to a unique label).

    Parameters
    ----------
    valid_size : float, default 0.5.
        From 0 to 1.
    n_splits : int or bool, optional (default=False).
        If False, n_splits is 1/valid_size (default : 1/0.5 = 2).
        If int : will iterate the number of times given in n_splits.
    random_state : integer or None, optional (default=None).
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is created with ``time.time()``.
    verbose : integer or False, optional (default=False).
        Controls the verbosity: the higher the value is, the more the messages are detailed.
    """

    def __init__(self,
                 valid_size=0.5,
                 n_splits=False,
                 random_state=None,
                 verbose=False):
        self.verbose = verbose

        self.crossvalidation = _cvc.groupCV

        if isinstance(valid_size, float):
            if valid_size > 1 or valid_size < 0:
                raise Exception('Percent must be between 0 and 1')
        else:
            raise Exception(
                'Percent must be between 0 and 1 and must be a float')
        if n_splits == False:
            n_splits = int(1 / valid_size)

        self.params = dict(
            valid_size=valid_size,
            n_splits=n_splits,
            random_state=random_state)

        _sampleSelection.__init__(self)


class LeaveOneSubGroupOut(_sampleSelection):
    """
    Generate a Cross-Validation by subgroup.

    Parameters
    ----------
    n_splits : int or bool, optional (default=False).
        If False : will iterate as many times as the smallest number of groups.
        If int : will iterate the number of times given in n_splits.
    random_state : integer or None, optional (default=None).
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is created with ``time.time()``.
    verbose : integer or False, optional (default=False).
        Controls the verbosity: the higher the value is, the more the messages are detailed.
    """

    def __init__(self,
                 n_splits=False,
                 random_state=None,
                 verbose=False):
        self.verbose = verbose

        self.crossvalidation = _cvc.groupCV

        self.params = dict(
            valid_size=1,
            n_splits=n_splits,
            random_state=random_state)
        _sampleSelection.__init__(self)


class SpatialLeaveAsideOut(_sampleSelection):
    """
    Generate a Cross-Validation using the farthest distance between the training and validation samples.

    Parameters
    ----------
    distance_matrix : array.
        Array got from function samplingMethods.getdistance_matrixForDistanceCV(inRaster,inVector)
    valid_size : float, default 0.5.
        The percentage of validaton to keep : from 0 to 1.
    n_splits : int or bool, optional (default=False).
        If False, n_splits is 1/valid_size (default : 1/0.5 = 2)
        If int : will iterate the number of times given in n_splits.
    random_state : integer or None, optional (default=None).
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is created with ``time.time()``.
    verbose : integer or False, optional (default=False).
        Controls the verbosity: the higher the value is, the more the messages are detailed.

    References
    ----------
    See "Combining ensemble modeling and remote sensing for mapping
    individual tree species at high spatial resolution" : https://doi.org/10.1016/j.foreco.2013.07.059.
    """

    def __init__(self,
                 distance_matrix=None,
                 valid_size=0.5,
                 n_splits=False,
                 random_state=None,
                 verbose=False):
        self.verbose = verbose

        self.crossvalidation = _cvc.distanceCV
        if n_splits == False:
            n_splits = int(1 / valid_size)

        self.params = dict(
            distance_matrix=distance_matrix,
            valid_size=valid_size,
            n_splits=n_splits,
            random_state=random_state)
        _sampleSelection.__init__(self)


class SpatialLeaveOneSubGroupOut(_sampleSelection):
    """
    Generate a Cross-Validation with Spatial Leave-One-Out method.

    Parameters
    ----------
    distance_matrix : numpy.ndarray, shape [n_samples, n_samples].
        Array got from function :fun:`museotoolbox.vector_tools.get_distance_matrix`
    distance_thresold : int.
        In pixels.
    distance_label : None or array.
        If array, got from function :fun:`museotoolbox.vector_tools.get_distance_matrix`
    n_splits : int or bool, optional (default=False).
        If False : will iterate as many times as the smallest number of groups.
        If int : will iterate the number of groups given in maxIter.
    random_state : integer or None, optional (default=None).
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is created with ``time.time()``.
    verbose : integer or False, optional (default=False).
        Controls the verbosity: the higher the value is, the more the messages are detailed.

    See also
    --------
    museotoolbox.vector_tools.get_distance_matrix : to get distance matrix and label.
    """

    def __init__(self,
                 distance_thresold,
                 distance_matrix,
                 distance_label,
                 n_splits=False,
                 random_state=None,
                 verbose=False):
        
        self.verbose = verbose

        self.crossvalidation = _cvc.distanceCV

        self.params = dict(
            distance_matrix=distance_matrix,
            distance_thresold=distance_thresold,
            distance_label=distance_label,
            SLOO=True,
            n_splits=n_splits,
            random_state=random_state)
        _sampleSelection.__init__(self)


class SpatialLeaveOneOut(_sampleSelection):
    """
    Generate a Cross-Validation with a stratified spatial Leave-One-Out method.

    Parameters
    ----------
    distance_matrix : array.
        Array got from function museotoolbox.vector_tools.get_distance_matrix(inRaster,inVector)
    distance_thresold : int.
        In pixels.
    n_splits : int or bool, optional (default=False).
        If False : will iterate as many times as the smallest number of groups.
        If int : will iterate the number of groups given in maxIter.
    random_state : int or None, default=None.
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is created with ``time.time()``.
    verbose : integer or False, optional (default=False).
        Controls the verbosity: the higher the value is, the more the messages are detailed.

    See also
    ---------
    museotoolbox.vector_tools.get_distance_matrix : to get distance matrix and label.
    
    References
    ----------
    See "Spatial leave‐one‐out cross‐validation for variable selection in the
    presence of spatial autocorrelation" : https://doi.org/10.1111/geb.12161.
    
    """

    def __init__(self,
                 distance_thresold=None,
                 distance_matrix=None,
                 n_splits=False,
                 random_state=None,
                 verbose=False,
                 **kwargs):

        self.verbose = verbose

        self.crossvalidation = _cvc.distanceCV

        self.params = dict(
            distance_matrix=distance_matrix,
            distance_thresold=distance_thresold,
            distance_label=False,
            minTrain=False,
            SLOO=True,
            n_splits=n_splits,
            random_state=random_state,
            **kwargs)
        _sampleSelection.__init__(self)


class RandomStratifiedKFold(_sampleSelection):
    """
    Generate a Cross-Validation with full random selection and Stratified K-Fold (same percentange per class).

    Parameters
    ----------
    n_splits : int, optional (default=2).
        Number of splits. 2 means 50% for each class at training and validation.
    n_repeats : integer or False, optional (default=False)
        If ``False``, will repeat n_splits once.
    random_state : integer or None, optional (default=None).
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is created with ``time.time()``.
    verbose : integer or False, optional (default=False).
        Controls the verbosity: the higher the value is, the more the messages are detailed.

    Example
    -------
    >>> from museotoolbox.cross_validation import StratifiedKFold
    >>> from museotoolbox import datasets
    >>> X,y = datasets.historicalMap(return_X_y=True)
    >>> SKF = StratifiedKFold(n_splits=2,random_state=12,verbose=False)
    >>> for tr,vl in SKF.split(X=X,y=y):
            print(tr,vl)
    [ 1600  1601  1605 ...,  9509  9561 10322] [ 3632  1988 11480 ..., 10321  9457  9508]
    [ 1599  1602  1603 ...,  9508  9560 10321] [ 3948 10928  3490 ..., 10322  9458  9561]
    """

    def __init__(self,
                 n_splits=2,
                 n_repeats=False,
                 random_state=None,
                 verbose=False):
        
        self.verbose = verbose

        valid_size = 1 / n_splits

        if n_repeats == False or n_repeats == 0:
            n_repeats = n_splits
        else:
            n_repeats = n_splits * n_repeats

        self.crossvalidation = _cvc.randomPerClass

        self.params = dict(
            valid_size=valid_size,
            random_state=random_state,
            n_splits=n_repeats)

        _sampleSelection.__init__(self)
