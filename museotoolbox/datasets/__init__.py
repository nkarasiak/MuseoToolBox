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
The :mod:`museotoolbox.datasets` module gathers available datasets for testing
`MuseoToolBox`.
"""
import os
__pathFile = os.path.dirname(os.path.realpath(__file__))


def historicalMap(return_X_y=False, return_X_y_g=False,
                  centroid=False, low_res=False):
    """
    Get a sample of a french Historical map made by the army (carte d'état-major).
    These maps are used to identify forest in the 1800's.

    Field of the vector containning the label class is `Class`.

    ===================   ==============
    Classes                            5
    Samples total                  12647
    Number of polygons                17
    Dimensionality                     3
    Features                     integer
    ===================   ==============


    Parameters
    -----------
    return_X_y : boolean, default=False.
        If True, returns ``(data, target)`` instead of a path of files.
    centroid : boolean, default=False.
        If True, return the path of the centroid for each feature.

    Returns
    -------
    raster,vector : list of str.
        Return path of raster and vector files.
    (data, target) : tuple if ``return_X_y`` is True

    References
    -----------
    https://github.com/nkarasiak/HistoricalMap

    Examples
    --------
    >>> X, y = getHistoricalMap(return_X_y=True)
    >>> X.shape, y.shape
    (12647, 3) (12647,)
    >>> raster,vector = getHistoricalMap()
    >>> raster
    /mnt/DATA/lib/MuseoToolBox/museotoolbox/datasets/historicalmap/map_compress.tif
    >>> vector
    /mnt/DATA/lib/MuseoToolBox/museotoolbox/datasets/historicalmap/train.gpkg
    """
    toReturn = []
    if low_res:
        raster = os.path.join(__pathFile, 'historicalmap/map_lowres.tif')
    else:
        raster = os.path.join(__pathFile, 'historicalmap/map_compress.tif')
    vector = os.path.join(__pathFile, 'historicalmap/train.gpkg')

    if return_X_y or return_X_y_g:
        from ..raster_tools import getSamplesFromROI
        if centroid:
            vector = os.path.join(
                __pathFile, 'historicalmap/train_centroid.gpkg')
        if return_X_y_g:
            X, y, g = getSamplesFromROI(raster, vector, 'Class', 'uniquefid')
            toReturn = (X, y, g)
        else:
            X, y = getSamplesFromROI(raster, vector, 'Class')
            toReturn = (X, y)
    else:
        toReturn.append(raster)
        toReturn.append(vector)
        if centroid:
            vectorCentroid = os.path.join(
                __pathFile, 'historicalmap/train_centroid.gpkg')
            toReturn.append(vectorCentroid)

    return toReturn
