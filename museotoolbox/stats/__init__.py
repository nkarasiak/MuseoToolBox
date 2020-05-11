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
The :mod:`museotoolbox.stats` module gathers stats functions and classes.
"""
import numpy as np
from museotoolbox.processing import RasterMath, extract_ROI, _add_vector_unique_fid

from scipy.ndimage.filters import generic_filter  # to compute moran


def zonal_stats(in_image, in_vector, unique_id, stats=[
                'mean', 'median', 'std'], verbose=False):
    """
    Extract zonal stats according to an predifined id.

    Parameters
    -----------
    in_image : str.
        Path of the raster file where the vector file will be rasterize.
    in_vector : str.
        Path of the vector file to rasterize.
    unique_id : str or False.
        If False, MuseoToolBox will create a field called 'uniquefid' using  thefunction :func:`_add_vector_unique_fid`.
    stats : list, optional (default=['mean','median','std']).
        str in list must be a function available from numpy.
        For example ['var'] will output the variance per band and per unique id.
    verbose : bool or int, optional (default=True).
        The higher is the int verbose, the more it will returns informations.


    Returns
    --------
    stats : np.ndarray
        Returns as many np.ndarray  as number of stats asked.
        Stats ordered by bands.

        For example (each line correspond to the unique_id ordered asc) :

        +-------------+---------------+-------------+
        | mean band_1 +  mean band_2  | mean band_3 |
        +-------------+---------------+-------------+
        | mean band_1 +  mean band_2  | mean band_3 |
        +-------------+---------------+-------------+

    Examples
    ---------
    >>> raster,vector = mtb.datasets.load_historical_data()
    >>> mean,var = mtb.stats.zonal_stats(raster,vector,'uniquefid',stats=['mean','var'])
    >>> mean.shape
    (17, 3)
    >>> mean[0,:] # mean of the first unique_id
    array([117.75219446, 109.80958812,  79.64213369])
    >>> var[0,:]
    array([1302.29983482, 1250.59980003, 1015.76659747])
    """
    if unique_id is False:
        _add_vector_unique_fid(in_vector, 'unique_id', verbose=verbose)
        unique_id = 'unique_id'

    X, y = extract_ROI(in_image, in_vector, unique_id)
    n_unique_id = len(np.unique(y))
    n_bands = X.shape[1]

    out_stats = [np.zeros([n_unique_id, n_bands]) for n in range(len(stats))]

    for idx_stat, stat in enumerate(stats):
        stat_function = getattr(__import__('numpy'), stat)
        for pos, label in enumerate(np.unique(y)):
            res = stat_function(X[np.where(y == label)], axis=0)
            out_stats[idx_stat][pos, :] = res

    return out_stats


class Moran:
    def __init__(self, in_image, in_image_mask=False,
                 lag=1, weights=False):
        """
        Compute Moran's I for raster.

        Parameters
        ----------
        in_image : str.
            A filename or path of a raster file.
            It could be any file that GDAL can open.
        lag : int, optional (default=1)
            The distance from the cell.
        weights :False or array-like, optional (default=False).
            Weights with the same shape as the square size.
        """

        self.scores = dict(lag=[], I=[], band=[], EI=[])
        self.lags = []

        rM = RasterMath(
            in_image,
            in_image_mask=in_image_mask,
            return_3d=True,
            verbose=False)

        for band, arr in enumerate(rM.read_band_per_band()):
            if isinstance(lag, int):
                lag = [lag]
            for l in lag:
                squareSize = l * 2 + 1
                footprint = np.ones((squareSize, squareSize))
                weights = np.zeros((footprint.shape[0], footprint.shape[1]))

                weights[:, 0] = 1
                weights[0, :] = 1
                weights[-1, :] = 1
                weights[:, -1] = 1

                n = arr.count()
                arr = arr.astype(np.float64)
                # convert masked to nan for generic_filter
                np.where(arr.mask, np.nan, arr.data)

                x_ = np.nanmean(arr)

                num = generic_filter(
                    arr,
                    self._compute_view_for_global_moran,
                    footprint=footprint,
                    mode='constant',
                    cval=np.nan,
                    extra_keywords=dict(
                        x_=x_,
                        footprint=footprint,
                        weights=weights,
                        transform='r'))

                den = (arr - x_)**2
                self.z = arr - x_

                # need to mask z/den/num/neighbors
                den[arr.mask] = np.nan
#                local = np.nansum(num) / np.nansum(den)
                l1 = np.where(arr.mask, np.nan, num)
                l2 = np.where(arr.mask, np.nan, den)
                local = np.nansum(l1) / np.nansum(l2)

                self.I = local

                self.EI = -1 / (n - 1)
                self.lags.append(l)
                self.scores['lag'].append(l)
                self.scores['band'].append(band + 1)
                self.scores['I'].append(self.I)
                self.scores['EI'].append(self.EI)

    def get_n_neighbors(self, array, footprint, weights):
        """
        Get number of neighbors according to your array and to a footprint.

        Parameters
        -----------
        array : array-like, shape = [X,Y]
            The input array.
        footprint : array-like, shape = [X,Y]
            The footprint array.
        weights : array-like, shape = [X,Y]
            The weight for each cell.

        Returns
        ---------
        w : int
            The number of neighbors.
        """
        b = np.reshape(
            array, (footprint.shape[0], footprint.shape[1])).astype(np.float64)
        xCenter = int((footprint.shape[0] - 1) / 2)
        yCenter = int((footprint.shape[1] - 1) / 2)
        b[xCenter, yCenter] = np.nan
        w = np.count_nonzero(~np.isnan(b))
        if w == 0:
            w = np.nan
        if weights is not False:
            weightsWindow = weights * footprint
            weightsNAN = weightsWindow[~np.isnan(b)]
            w = np.nansum(weightsNAN)
        return w

    def _compute_view_for_global_moran(
            self, a, x_, footprint, weights, transform='r'):
        xSize, ySize = footprint.shape
        a = np.reshape(a, (xSize, ySize))

        xCenter = int((xSize - 1) / 2)
        yCenter = int((ySize - 1) / 2)

        xi = a[xCenter, yCenter]
        if np.isnan(xi):
            return np.nan
        else:
            a[xCenter, yCenter] = np.nan
            w = np.count_nonzero(~np.isnan(a))
            if w == 0:
                num = np.nan
            else:
                w = np.copy(weights)
                if transform == 'r':
                    w = 1 / np.count_nonzero(~np.isnan(a))

                num = np.nansum(w * (a - x_) * (xi - x_))

        return num


def commission_omission(table):
    """
    Compute commission and omission from a confusion matrix

    Parameters
    ----------
    table : array.
        The confusion matrix (same number of lines and columns)

    Returns
    -------
    com : commissions (list)
    om : omission (list)
    """
    com, om = [[], []]
    for i in range(table.shape[0]):
        com.append((np.sum(table[i, :]) - table[i, i]
                    ) / float(np.sum(table[i, :])) * 100)
        om.append((np.sum(table[:, i]) - table[i, i]) /
                  float(np.sum(table[:, i])) * 100)
    return com, om


class ComputeConfusionMatrix:
    def __init__(self, yp, yr, kappa=False, OA=False, F1=False):
        """
        Compute confusion matrix given label predicted and label reality.

        Parameters
        ----------
        yp : array.
            The label predicted.
        yr : array.
            The label truth.
        kappa : bool, default False.
            If true, computes kappa.
        OA : bool, default False.
            If True, computes Overall Accuracy.
        F1 : bool, default False.
            If True, computes F1-Score per class.
        """
        # Initialization
        if isinstance(yp, list):
            yp = np.asarray(yp)
            yr = np.asarray(yr)
        n = yp.size
        C = np.amax((int(yr.max()), int(yp.max())))
        self.confusion_matrix = np.zeros((C, C), dtype=np.int64)

        # Compute confusion matrix
        for i in range(n):
            self.confusion_matrix[yp[i].astype(
                np.int64) - 1, yr[i].astype(np.int64) - 1] += 1

        # Compute overall accuracy
        if OA:
            self.OA = np.sum(np.diag(self.confusion_matrix)) / n

        # Compute Kappa
        if kappa:
            nl = np.sum(self.confusion_matrix, axis=1)
            nc = np.sum(self.confusion_matrix, axis=0)
            self.Kappa = ((n**2) * np.sum(np.diag(self.confusion_matrix)
                                          ) / n - np.sum(nc * nl)) / (n**2 - np.sum(nc * nl))

        #
        if F1:
            F1 = []
            for label in range(self.confusion_matrix.shape[0]):
                TP = self.confusion_matrix[label, label]
                #TN = np.sum(sp.diag(currentCsv))-currentCsv[label,label]
                FN = np.sum(
                    self.confusion_matrix[:, label]) - self.confusion_matrix[label, label]
                FP = np.sum(
                    self.confusion_matrix[label, :]) - self.confusion_matrix[label, label]
                denum = (2 * TP + FP + FN)
                if denum != 0:
                    F1.append(2 * TP / (2 * TP + FP + FN))
                self.F1 = F1


def retrieve_y_from_confusion_matrix(confusion_matrix):
    """
    Retrieve y_predict and y_truth from confusion matrix.

    Parameters
    -----------
    confusion_matrix : nd-array of shape [number of labels, number of labels]
        The confusion matrix

    Returns
    --------
    yt,yp : two nd-array of shape [sum of confusion matrix,]

    """
    confusion_matrix = np.asarray(confusion_matrix)

    yp = []
    for j in range(confusion_matrix.shape[0]):
        for i in range(confusion_matrix.shape[1]):
            yp.extend([i + 1] * confusion_matrix[j, i])
    yp = np.asarray(yp)
    yt = [[i + 1] * np.sum(confusion_matrix[i, :])
          for i in range(confusion_matrix.shape[0])]
    yt = np.asarray([item for sublist in yt for item in sublist])

    return yt, yp
