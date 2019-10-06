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
from .. import vector_tools, raster_tools
from ..internal_tools import progressBar

import sys
import os
import numpy as np
from osgeo import gdal, ogr


class sampleExtraction:
    def __init__(
            self,
            inRaster,
            inVector,
            outVector,
            uniqueFID=None,
            bandPrefix=None,
            verbose=1):
        """
        Extract centroid from shapefile according to the raster, and extract band value if bandPrefix is given.

        This script is available via terminal by entering : `mtb_sampleExtraction`.

        Parameters
        ----------
        inRaster : str.
            Raster path.
        inVector : str.
            Vector path.
        outVector : str.
            Outvector. Extension will be used to select driver. Please use ['sqlite','shp','netcdf','gpx'].
        uniqueFID : str, default None.
            If None, will add a field called 'uniquefid' in the output vector.
        bandPrefix : str, default None.
            If bandPrefix (e.g. 'band'), will extract values from raster.
        """
        self.__verbose = verbose
        if uniqueFID is None:
            uniqueFID = 'uniquefid'
            if verbose:
                print("Adding 'uniquefid' field to the original vector.")
            vector_tools.addUniqueIDForVector(
                inVector, uniqueFID, verbose=verbose)

        if verbose:
            print("Extract values from raster...")
        X, Y, coords = raster_tools.getSamplesFromROI(
            inRaster, inVector, uniqueFID, getCoords=True, verbose=verbose)

        geoTransform = gdal.Open(inRaster).GetGeoTransform()

        centroid = [self.__pixelLocationToCentroidGeom(
            coord, geoTransform) for coord in coords]
        # init outLayer
        if np.issubdtype(X.dtype, np.integer):
            try:
                dtype = ogr.OFTInteger64
            except BaseException:
                dtype = ogr.OFTInteger
        else:
            dtype = ogr.OFTReal
        outLayer = createPointLayer(
            inVector, outVector, uniqueFID, dtype=dtype)
        if self.__verbose:
            outLayer.addTotalNumberOfPoints(len(centroid))

        if verbose:
            print("Adding each centroid to {}...".format(outVector))
        for idx, xy in enumerate(centroid):
            try:
                curY = Y[idx][0]
            except BaseException:
                curY = Y[idx]
            if curY != 0:
                if bandPrefix is None:
                    outLayer.addPointToLayer(xy, curY)
                else:
                    outLayer.addPointToLayer(xy, curY, X[idx], bandPrefix)

        outLayer.closeLayers()

    def __pixelLocationToCentroidGeom(self, coords, geoTransform):
        """
        Convert XY coords into the centroid of a pixel

        Parameters
        --------
        coords : arr or list.
            X is coords[0], Y is coords[1].
        geoTransform : list.
            List got from gdal.Open(inRaster).GetGeoTransform() .
        """
        newX = geoTransform[1] * (coords[0] + 0.5) + \
            geoTransform[0]  # (coords[0]+0.5)*geoTransform[1]+geoTransform[0]# (coords[0]+0.5)*geoTransform[1]+geoTransform[0]
        # (coords[1]+0.5)*geoTransform[5]+geoTransform[3] # (coords[1]+0.5)*geoTransform[5]+geoTransform[3]
        newY = geoTransform[5] * (coords[1] + 0.5) + geoTransform[3]
        return [newX, newY]


class createPointLayer:
    def __init__(self, inVector, outVector, uniqueIDField,
                 dtype=ogr.OFTInteger, verbose=1):
        """
        Create a vector layer as point type.

        Parameters
        -------
        inVector : str.
            Vector to copy fields and spatial reference.
        outVector : str.
            According to the file extension, will found the good driver from OGR.
        uniqueIDField : str.
            Field containing the unique ID for each feature.
        bandPrefix : None or str.
            If str, if the prefix of each value for the number of bands of your image. Used to train models from classifier.

        Functions
        -------
        addTotalNumberOfPoints(nSamples): int.
            Will generate progress bar.
        addPointToLayer(coords): list,arr.
            coords[0] is X, coords[1] is Y.
        closeLayer():
            Close the layer.
        """
        self.__verbose = verbose
        self.__dtype = dtype
        # load inVector
        self.inData = ogr.Open(inVector, 0)
        self.inLyr = self.inData.GetLayerByIndex(0)
        srs = self.inLyr.GetSpatialRef()

        # create outVector
        self.driverName = vector_tools.getDriverAccordingToFileName(outVector)
        driver = ogr.GetDriverByName(self.driverName)
        self.outData = driver.CreateDataSource(outVector)

        # finish  outVector creation
        self.outLyr = self.outData.CreateLayer('centroid', srs, ogr.wkbPoint)
        self.outLyrDefinition = self.outLyr.GetLayerDefn()

        # initialize variables
        self.idx = 0
        self.lastPosition = 0
        self.nSamples = None
        if self.driverName == 'SQLITE' or self.driverName == 'GPKG':
            self.outLyr.StartTransaction()

        self.uniqueIDField = uniqueIDField

        # Will generate uniqueIDandFID when copying vector
        self.uniqueIDandFID = False
        self.addBand = False

    def __addBandsValue(self, bandPrefix, nBands):
        """
        Parameters
        -------
        bandPrefix : str.
            Prefix for each band (E.g. 'band')
        nBands : int.
            Number of band to save.
        """
        self.nBandsFields = []
        for b in range(nBands):
            field = bandPrefix + str(b)
            self.nBandsFields.append(field)
            self.outLyr.CreateField(ogr.FieldDefn(field, self.__dtype))
        self.addBand = True

    def addTotalNumberOfPoints(self, nSamples):
        """
        Adding the total number of points will show a progress bar.

        Parameters
        --------
        nSamples : int.
            The number of points to be added (in order to have a progress bar. Will not affect the processing if bad value is put here.)
        """
        self.nSamples = nSamples
        self.pb = progressBar(nSamples, 'Adding points... ')

    def addPointToLayer(
            self,
            coords,
            uniqueIDValue,
            bandValue=None,
            bandPrefix=None):
        """
        Parameters
        -------
        coords : list, or arr.
            X is coords[0], Y is coords[1]
        uniqueIDValue : int.
            Unique ID Value to retrieve the value from fields
        bandValue : None, or arr.
            If array, should have the same size as the number of bands defined in addBandsValue function.
        """
        if self.__verbose:
            if self.nSamples:
                currentPosition = int(self.idx + 1)
                if currentPosition != self.lastPosition:
                    self.pb.addPosition(self.idx + 1)
                    self.lastPosition = currentPosition

        if self.uniqueIDandFID is False:
            self.__updateArrAccordingToVector__()

        # add Band to list of fields if needed
        if bandValue is not None:
            if bandPrefix is None:
                raise Warning(
                    'Please, define a bandPrefix value to save bands value into the vector file.')
            if self.addBand is False:
                self.__addBandsValue(bandPrefix, bandValue.shape[0])

        point = ogr.Geometry(ogr.wkbPoint)
        point.SetPoint(0, coords[0], coords[1])
        featureIndex = self.idx
        feature = ogr.Feature(self.outLyrDefinition)
        feature.SetGeometry(point)
        feature.SetFID(featureIndex)

        # Retrieve inVector FID
        try:
            FID = self.uniqueFIDs[np.where(np.asarray(
                self.uniqueIDs, dtype=np.int) == int(uniqueIDValue))[0][0]]
        except BaseException:
            print(uniqueIDValue)

        featUpdates = self.inLyr.GetFeature(int(FID))
        for f in self.fields:
            if f != 'ogc_fid':
                feature.SetField(f, featUpdates.GetField(f))
                if self.addBand is True:
                    for idx, f in enumerate(self.nBandsFields):
                        feature.SetField(f, int(bandValue[idx]))

        self.outLyr.CreateFeature(feature)
        self.idx += 1

    def __updateArrAccordingToVector__(self):
        """
        Update outVector layer by adding field from inVector.
        Store ID and FIDs to find the same value.
        """
        self.uniqueIDs = []
        self.uniqueFIDs = []
        currentFeature = self.inLyr.GetNextFeature()
        self.fields = [
            currentFeature.GetFieldDefnRef(i).GetName() for i in range(
                currentFeature.GetFieldCount())]
        # Add input Layer Fields to the output Layer
        layerDefinition = self.inLyr.GetLayerDefn()

        for i in range(len(self.fields)):
            fieldDefn = layerDefinition.GetFieldDefn(i)
            self.outLyr.CreateField(fieldDefn)

        self.inLyr.ResetReading()
        for feat in self.inLyr:
            uID = feat.GetField(self.uniqueIDField)
            uFID = feat.GetFID()
            self.uniqueIDs.append(uID)
            self.uniqueFIDs.append(uFID)
        self.uniqueIDandFID = True

    def closeLayers(self):
        """
        Once work is done, close all layers.
        """
        if self.driverName == 'SQLITE' or self.driverName == 'GPKG':
            self.outLyr.CommitTransaction()
        self.inData.Destroy()
        self.outData.Destroy()


def main(argv=None, apply_config=True):
    import argparse
    if len(sys.argv) == 1:
        prog = os.path.basename(sys.argv[0])
        print(sys.argv[0] + ' [options]')
        print("Help : ", prog, " --help")
        print("or : ", prog, " -h")
        print(
            2 *
            ' ' +
            "example 1 : ",
            prog,
            " -in raster.tif -vec roi.sqlite -out vector.sqlite -outfield.prefix.name band_ ")
        print(
            2 *
            ' ' +
            "example 2 : ",
            prog,
            " -in raster.tif -vec roi.sqlite -out vector.sqlite -field ogc_fid")
        sys.exit(-1)

    else:
        usage = "usage: %prog [options] "
        parser = argparse.ArgumentParser(
            description="From points or polygons, extraction each pixel centroid and extract values from raster.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "-in",
            "--image",
            dest="inRaster",
            action="store",
            help="Image to extract values from and to generate centroid from output vector",
            required=True)

        parser.add_argument(
            "-vec",
            "--vector",
            dest="inVector",
            action="store",
            help="Vector to fill with raster values",
            required=True,
            type=str)

        parser.add_argument(
            "-out",
            "--outvector",
            dest="outVector",
            action="store",
            help="Vector to save (sqlite extension if possible)",
            required=True,
            type=str)

        parser.add_argument(
            "-field",
            "--uniqueField",
            dest="uniqueFID",
            action="store",
            help="Unique field per feature. If no field, will create an 'uniquefid' field in the original shapefile.",
            required=False,
            default=None)

        parser.add_argument(
            "-outfield.prefix.name",
            "--outField",
            dest="bandPrefix",
            action="store",
            help="Prefix name to save the values from the raster. E.g 'band_'.",
            required=False,
            default=None)

        args = parser.parse_args()

        sampleExtraction(
            inRaster=args.inRaster,
            inVector=args.inVector,
            outVector=args.outVector,
            uniqueFID=args.uniqueFID,
            bandPrefix=args.bandPrefix)


if __name__ == "__main__":
    sys.exit(main())
