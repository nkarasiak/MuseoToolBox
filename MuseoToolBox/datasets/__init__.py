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

import os
__pathFile = os.path.dirname(os.path.realpath(__file__))


def getHistoricalMap(centroid=False):
    """
    Get French Historical map dataset.
    
    Parameters
    -----------
    centroid : bool.
        If true, return the path of the centroid for each feature.
        
    Returns
    -------
    list of files : list.
        Raster and vector path, and the centroid if asked.
    """
    toReturn = []
    
    raster = os.path.join(__pathFile, 'historicalmap/map_compress.tif')
    vector = os.path.join(__pathFile, 'historicalmap/train.gpkg')
    
    toReturn.append(raster)
    toReturn.append(vector)
    if centroid:
        vectorCentroid = os.path.join(__pathFile, 'historicalmap/train_centroid.gpkg')
        toReturn.append(vectorCentroid)
    
    return toReturn
