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
from museotoolbox.geo_tools import RasterMath

from scipy.ndimage.filters import generic_filter  # to compute moran


class Moran:
    def __init__(self, in_image, in_image_mask=False, transform='r',
                 lag=1, weights=False, intermediate_lag=True):
        """
        Compute Moran's I for raster.

        Parameters
        ----------
        in_image : str.
            A filename or path of a raster file.
            It could be any file that GDAL can open.
        in_image_mask   :   str.
                            Path to mask raster, default False.
        transform       :   str, optional (default='r')
            default optional is row-standardized (transform='r').
            for binary transfrom (transform='b').
        lag : int, optional (default=1)
            The distance from the cell.
        weights :False or array-like, optional (default=False).
            Weights with the same shape as the square size.
        intermediate_lag :  bool, optional (default=True).
            Use all pixel values inside the specified lag.
            If `intermediate_lag` is set to False, only the
            pixels at the specified range will be kept for computing the statistics.
        """

        self.scores = dict(I=[], band=[], EI=[])
        self.lags = []
        if isinstance(in_image, (np.ma.core.MaskedArray, np.ndarray)):
            arr = in_image
        else:
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

                if not intermediate_lag:
                    weights[:, 0] = 1
                    weights[0, :] = 1
                    weights[-1, :] = 1
                    weights[:, -1] = 1
                else:
                    weights[:, :] = 1

                if transform == 'b' and band == 0:
                    neighbors = generic_filter(
                        arr,
                        self.get_n_neighbors,
                        footprint=footprint,
                        mode='constant',
                        cval=np.nan,
                        extra_keywords=dict(
                            footprint=footprint,
                            weights=weights))
                    if not np.all(arr.mask == False):
                        neighbors[arr.mask] = np.nan

                n = arr.count()
                arr = arr.astype(np.float64)
                # convert masked to nan for generic_filter
                arr.data[arr.mask] = np.nan

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
                        transform=transform))

                den = (arr - x_)**2
                self.z = arr - x_

                # need to mask z/den/num/neighbors
                den[arr.mask] = np.nan
                local = np.nansum(num) / np.nansum(den)
                if transform == 'b':
                    self.I = (n / np.nansum(neighbors)) * local
                else:
                    self.I = local
                self.EI = -1 / (n - 1)
                self.lags.append(l)
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
                if weights is False:
                    w = np.copy(footprint)
                else:
                    w = np.copy(weights)
                if transform == 'r':
                    w /= np.count_nonzero(~np.isnan(a))

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
            self.Kappa = ((n**2) * np.sum(np.diag(self.confusion_matrix)) / n - np.sum(nc * nl)) / \
                (n**2 - np.sum(nc * nl))

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
