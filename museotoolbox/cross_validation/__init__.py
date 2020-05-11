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
import numpy as np
from . import _sample_selection


def train_test_split(cv, X, y, random_state=False, **kwargs):
    """
    Split arrays into random train and test subsets according to your choosen cross_validation method.

    Quick utility that wraps input validation and next(ShuffleSplit().split(X, y)) and application to input data into a single call for splitting (and optionally subsampling) data in a oneliner.

    Parameters
    -----------
    cv : cross-validation function.
        Allowed function from museotoolbox as scikit-learn.
    X : array-like, shape (n_samples, n_features), optional
        Training data, where n_samples is the number of samples
        and n_features is the number of features.
    y : array-like, of length n_samples
        The target variable for supervised learning problems.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Examples
    ---------
    import numpy as np
    import museotoolbox as mtb

    X, y = np.arange(10).reshape((5, 2)), range(5)
    cv = mtb.cross_validation.LeaveOneOut(random_state=42)
    X_train, y_train, X_test, y_test = mtb.cross_validation.train_test_split(cv,X,y)
    """
    # X_train, X_test = [np.asarray([],dtype=X.dtype).reshape(-1,X.shape[-1])]*2 # empty X
    # y_train, y_test = [np.asarray([],dtype=np.int64)]*2 # empty y
    if y.ndim == 2:
        y = y.flatten()

    for tr, vl in cv.split(X, y, **kwargs):
        X_train = X[tr, ...]
        y_train = y[tr]
        X_test = X[vl, ...]
        y_test = y[vl]
        if 'groups' in kwargs:
            g = kwargs['groups']
            if g.ndim == 2:
                g = y.flatten()
            g_train = g[tr, ...]
            g_test = g[vl, ...]
        break  # only the first fold is needed

    if 'groups' in kwargs:
        return X_train, X_test, y_train, y_test, g_train, g_test
    else:
        return X_train, X_test, y_train, y_test


class LeaveOneOut(_sample_selection._cv_manager):
    """
    Generate a Cross-Validation using a Stratified Leave One Out.
    Note : :class:`~LeaveOneOut` is equivalent to :class:`~museotoolbox.cross_validation.RandomStratifiedKFold` with ``valid_size=1`` and ``n_splits=False``.

    Parameters
    ----------
    n_repeats : int or bool, optional (default=False).
        If False : will iterate as many times as the smallest class.
        If int : will iterate the number of times given in n_splits.
    random_state : integer or None, optional (default=False).
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is created with ``time.time()``.
    verbose : integer or False, optional (default=False).
        Controls the verbosity: the higher the value is, the more the messages are detailed.
    """

    def __init__(self,
                 n_repeats=False,
                 random_state=False,
                 verbose=False):

        super().__init__(
            _sample_selection.randomPerClass,
            valid_size=1,
            n_repeats=n_repeats,
            random_state=random_state,
            verbose=verbose)


class LeavePSubGroupOut(_sample_selection._cv_manager):
    """
    Generate a Cross-Validation using subgroup (each group belong to a unique label).

    Parameters
    ----------
    valid_size : float, default 0.5.
        From 0 to 1.
    n_repeats : int or bool, optional (default=False).
        If False, n_splits is 1/valid_size (default : 1/0.5 = 2).
        If int : will iterate the number of times given in n_splits.
    random_state : integer or None, optional (default=False).
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is created with ``time.time()``.
    verbose : integer or False, optional (default=False).
        Controls the verbosity: the higher the value is, the more the messages are detailed.
    """

    def __init__(self,
                 valid_size=0.5,
                 n_repeats=False,
                 random_state=False,
                 verbose=False):

        if isinstance(valid_size, float):
            if valid_size > 1 or valid_size < 0:
                raise ValueError('Percent must be between 0 and 1')
        else:
            raise ValueError(
                'Percent must be between 0 and 1 and must be a float')
        if not n_repeats:
            n_repeats = int(1 / valid_size)

        super().__init__(
            _sample_selection.groupCV,
            valid_size=valid_size,
            n_repeats=n_repeats,
            random_state=random_state,
            verbose=verbose)


class LeaveOneSubGroupOut(_sample_selection._cv_manager):
    """
    Generate a Cross-Validation by subgroup.

    Parameters
    ----------
    n_repeats : int or bool, optional (default=False).
        If False : will iterate as many times as the smallest number of groups.
        If int : will iterate the number of times given in n_splits.
    random_state : integer or None, optional (default=False).
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is created with ``time.time()``.
    verbose : integer or False, optional (default=False).
        Controls the verbosity: the higher the value is, the more the messages are detailed.
    """

    def __init__(self,
                 n_repeats=False,
                 random_state=False,
                 verbose=False):

        super().__init__(
            _sample_selection.groupCV,
            valid_size=1,
            n_repeats=n_repeats,
            random_state=random_state,
            verbose=verbose)


class SpatialLeaveAsideOut(_sample_selection._cv_manager):
    """
    Generate a Cross-Validation using the farthest distance between the training and validation samples.

    Parameters
    ----------
    distance_matrix : numpy.ndarray, shape [n_samples, n_samples].
        Array got from function samplingMethods.getdistance_matrixForDistanceCV(inRaster,inVector)
    valid_size : float, default 0.5.
        The percentage of validaton to keep : from 0 to 1.
    n_repeats : int or bool, optional (default=False).
        If False, n_repeats is 1/valid_size (default : 1/0.5 = 2)
        If int : will iterate the number of times given in n_repeats.
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
                 distance_matrix,
                 valid_size=0.5,
                 n_repeats=False,
                 random_state=False,
                 verbose=False):

        super().__init__(
            _sample_selection.distanceCV,
            distance_matrix=distance_matrix,
            valid_size=valid_size,
            n_repeats=n_repeats,
            random_state=random_state,
            verbose=verbose)


class SpatialLeaveOneSubGroupOut(_sample_selection._cv_manager):
    """
    Generate a Cross-Validation with Spatial Leave-One-Out method.

    Parameters
    ----------
    distance_matrix : numpy.ndarray, shape [n_samples, n_samples].
        Array got from function :func:`museotoolbox.vector_tools.get_distance_matrix`
    distance_thresold : int.
        In pixels.
    distance_label : None or array.
        If array, got from function :func:`museotoolbox.vector_tools.get_distance_matrix`
    random_state : integer or None, optional (default=None).
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is created with ``time.time()``.
    verbose : integer or False, optional (default=False).
        Controls the verbosity: the higher the value is, the more the messages are detailed.

    See also
    --------
    museotoolbox.processing.get_distance_matrix : to get distance matrix and label.
    """

    def __init__(self,
                 distance_thresold,
                 distance_matrix,
                 distance_label,
                 random_state=False,
                 verbose=False):

        super().__init__(
            _sample_selection.distanceCV,
            distance_matrix=distance_matrix,
            distance_thresold=distance_thresold,
            distance_label=distance_label,
            random_state=random_state,
            verbose=verbose)


class SpatialLeaveOneOut(_sample_selection._cv_manager):
    """
    Generate a Cross-Validation with a stratified spatial Leave-One-Out method.

    Parameters
    ----------
    distance_matrix : numpy.ndarray, shape [n_samples, n_samples].
        Array got from function museotoolbox.vector_tools.get_distance_matrix(inRaster,inVector)
    distance_thresold : int.
        In pixels.
    n_repeats: int or False, optional (default=False).
        If False : will iterate as many times as the smallest number of groups.
        If int : will iterate the number of times specified.
    random_state : int or False, optional (default=False).
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
                 n_repeats=False,
                 n_splits=False,
                 random_state=False,
                 verbose=False,
                 **kwargs):

        super().__init__(
            _sample_selection.distanceCV,
            distance_matrix=distance_matrix,
            distance_thresold=distance_thresold,
            distance_label=False,
            valid_size=1,
            n_repeats=n_repeats,
            n_splits=n_splits,
            random_state=random_state,
            verbose=verbose,
            **kwargs)


class RandomStratifiedKFold(_sample_selection._cv_manager):
    """
    Generate a Cross-Validation with full random selection and Stratified K-Fold (same percentange per class).

    Parameters
    ----------
    n_splits : int, optional (default=2).
        Number of splits. 2 means 50% for each class at training and validation.
    n_repeats : integer or False, optional (default=False)
        If False, will repeat n_splits once.
    valid_size : int or False, optional (default=False).
        If False, valid size is ``1 / n_splits``.
    random_state : integer or None, optional (default=False).
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is created with ``time.time()``.
    verbose : integer or False, optional (default=False).
        Controls the verbosity: the higher the value is, the more the messages are detailed.

    Example
    -------
    >>> from museotoolbox.cross_validation import RandomStratifiedKFold
    >>> from museotoolbox import datasets
    >>> X,y = datasets.load_historical_data(return_X_y=True)
    >>> RSK = RandomStratifiedKFold(n_splits=2,random_state=12,verbose=False)
    >>> for tr,vl in RSK.split(X=X,y=y):
            print(tr,vl)
    [ 1600  1601  1605 ...,  9509  9561 10322] [ 3632  1988 11480 ..., 10321  9457  9508]
    [ 1599  1602  1603 ...,  9508  9560 10321] [ 3948 10928  3490 ..., 10322  9458  9561]
    """

    def __init__(self,
                 n_splits=2,
                 n_repeats=False,
                 valid_size=False,
                 random_state=False,
                 verbose=False):

        if valid_size is False:
            valid_size = 1 / n_splits

        if n_repeats == False or n_repeats == 0:
            n_repeats = n_splits
        else:
            n_repeats = n_splits * n_repeats

        super().__init__(
            _sample_selection.randomPerClass,
            valid_size=valid_size,
            random_state=random_state,
            n_repeats=n_repeats,
            verbose=verbose)
