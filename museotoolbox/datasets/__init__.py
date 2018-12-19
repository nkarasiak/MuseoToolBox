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
The :mod:`museotoolbox.datasets` module gathers available datasets for testing
`Museo ToolBox`.
"""
import os
__pathFile = os.path.dirname(os.path.realpath(__file__))


def getHistoricalMap(return_X_y=False, centroid=False):
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
    https://github.com/lennepkade/HistoricalMap

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

    raster = os.path.join(__pathFile, 'historicalmap/map_compress.tif')
    vector = os.path.join(__pathFile, 'historicalmap/train.gpkg')

    if return_X_y:
        from ..rasterTools import getSamplesFromROI
        if centroid:
            vector = os.path.join(
                __pathFile, 'historicalmap/train_centroid.gpkg')
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