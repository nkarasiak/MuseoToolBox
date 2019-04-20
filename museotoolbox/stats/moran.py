#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:58:35 2019

@author: nicolas
"""
import numpy as np
from scipy.ndimage.filters import generic_filter
from time import time

from ..raster_tools import rasterMath


class Moran:
    """
    Compute Moran's I for raster.

    Parameters
    ----------
    inRaster : str.
        Path.
    inMaskRaster : str, default False.
        lag : int or
    transform : str.
        'r' or 'b'.
    weights : False or array.
        Weights (same shape as the square size).
    intermediate_lag : boolean, default True.
        Use all pixel values inside the specified lag.

        If `intermediate_lag` is set to False, only the pixels at the specified
        range will be kept for computing the statistics.


    """

    def __init__(self, inRaster, inMaskRaster=False, transform='r',
                 lag=1, weights=False, intermediate_lag=True):

        self.scores = dict(I=[], band=[], EI=[])
        if isinstance(inRaster, (np.ma.core.MaskedArray, np.ndarray)):
            arr = inRaster
        else:
            rM = rasterMath(
                raster,
                inMaskRaster=mask,
                return_3d=True,
                verbose=False)

        for band, arr in enumerate(rM.readBandPerBand()):
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
                        self.getNNeighbors,
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
                    self.__computeViewForGlobanMoran,
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
                self.scores['band'].append(band + 1)
                self.scores['I'].append(self.I)
                self.scores['EI'].append(self.EI)

    def getNNeighbors(self, array, footprint, weights):

        b = np.reshape(array, (footprint.shape[0], footprint.shape[1]))
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

    def __computeViewForGlobanMoran(
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
