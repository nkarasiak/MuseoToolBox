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
The :mod:`museotoolbox.raster_tools` module gathers raster functions.
"""

from __future__ import absolute_import, print_function

import gdal
import numpy as np
import os
import tempfile

from ..internal_tools import progressBar, pushFeedback
from ..vector_tools import sampleExtraction
from matplotlib import pyplot as plt


def rasterMaskFromVector(inVector, inRaster, outRaster, invert=False):
    """
    Create a raster mask where polygons/point are pixels to keep.

    Parameters
    ----------
    inVector : str.
        Path of the vector file to rasterize.
    inRaster : str.
        Path of the raster file where the vector file will be rasterize.
    outRaster : str.
        Path of the file (.tif) to create.
    Returns
    -------
    None

    Examples
    --------
    """
    rasterize(
        inRaster,
        inVector,
        None,
        outRaster,
        invert=invert,
        gdt=gdal.GDT_Byte)


def getGdalDTFromMinMaxValues(maxValue, minValue=0):
    """
    Return the Gdal DataType according the minimum or the maximum value.

    Parameters
    ----------
    maxValue : int/float.
        The maximum value needed.
    minValue : int/float, default 0.
        The minimum value needed.

    Returns
    -------
    gdalDT : integer.

    Examples
    ---------
    >>> getGdalDTFromMinMaxValues(260)
    2
    >>> getGdalDTFromMinMaxValues(16)
    1
    >>> getGdalDTFromMinMaxValues(16,-260)
    3
    """
    maxAbsValue = np.amax(np.abs([maxValue, minValue]))

    # if values are integer
    if isinstance(maxAbsValue, (int, np.integer)):
        if minValue >= 0:
            if maxValue <= 255:
                gdalDT = gdal.GDT_Byte
            elif maxValue > 255 and maxValue <= 65535:
                gdalDT = gdal.GDT_UInt16
            elif maxValue >= 65535:
                gdalDT = gdal.GDT_UInt32
        elif minValue < 0:
            if minValue > -65535:
                gdalDT = gdal.GDT_Int16
            else:
                gdalDT = gdal.GDT_Int32

    # if values are float
    if isinstance(maxAbsValue, float):
        if maxAbsValue <= +3.4E+38:
            gdalDT = gdal.GDT_Float32
        else:
            gdalDT = gdal.GDT_Float64

    return gdalDT


def convertGdalAndNumpyDataType(gdalDT=None, numpyDT=None):
    """
    Return the datatype from gdal to numpy or from numpy to gdal.

    Parameters
    -----------
        gdalDT : int
            gdal datatype from src_dataset.GetRasterBand(1).DataType.
        numpyDT : str
            str from array.dtype.name.

    Returns
    --------
        dt : the data type (int for Gdal or type for numpy)

    Examples
    ---------
    >>> convertGdalAndNumpyDataType(gdal.GDT_Int16)
    numpy.int16
    >>> convertGdalAndNumpyDataType(gdal.GDT_Float64)
    numpy.float64
    >>> convertGdalAndNumpyDataType(numpyDT=np.array([],dtype=np.int16).dtype.name)
    3
    >>> convertGdalAndNumpyDataType(numpyDT=np.array([],dtype=np.float64).dtype.name)
    7
    """
    from osgeo import gdal_array

    NP2GDAL_CONVERSION = {
        "uint8": 1,
        "int8": 1,
        "uint16": 2,
        "int16": 3,
        "uint32": 4,
        "int32": 5,
        "float32": 6,
        "float64": 7,
        "complex64": 10,
        "complex128": 11,
        "int64": 5
    }

    if numpyDT is None:
        code = gdal_array.GDALTypeCodeToNumericTypeCode(gdalDT)
    else:

        code = NP2GDAL_CONVERSION[numpyDT]
        if numpyDT == 'int64':
            print(
                'Warning : Numpy type {} is not recognized by gdal. Will use int32 instead'.format(numpyDT))
    return code


def convertGdalDataTypeToOTB(gdalDT):
    """
    Convert Gdal DataType to OTB str format.

    Parameters
    ----------
    gdalDT : int
        gdal Datatype (integer value)

    Returns
    ----------
    str format of OTB datatype
        availableCode = uint8/uint16/int16/uint32/int32/float/double

    Examples
    ---------
    >>> convertGdalDataTypeToOTB(gdal.GDT_Float32)
    'float'
    >>> convertGdalDataTypeToOTB(gdal.GDT_Byte)
    'uint8'
    >>> convertGdalDataTypeToOTB(gdal.GDT_UInt32)
    'uint32'
    >>> convertGdalDataTypeToOTB(gdal.GDT_CFloat64)
    'cdouble'
    """
    # uint8/uint16/int16/uint32/int32/float/double/cint16/cint32/cfloat/cdouble
    code = [
        'uint8',
        'uint8',
        'uint16',
        'int16',
        'uint32',
        'int32',
        'float',
        'double',
        'cint16',
        'cint32',
        'cfloat',
        'cdouble']
    if gdalDT > len(code):
        otbDT = ('cdouble')
    else:
        otbDT = code[gdalDT]

    return otbDT


def getSamplesFromROI(inRaster, inVector, *fields, **kwargs):
    """
    Get the set of pixels given the thematic map. Both map should be of same size. Data is read per block.

    Initially written by Mathieu Fauvel, improved by Nicolas Karasiak.

    Parameters
    -----------
    inRaster: str.
        the name of the raster file, could be any file that GDAL can open
    inVector: str.
        the name of the thematic image: each pixel whose values is greater than 0 is returned
    *fields : str.
        Each field to extract label/value from.
    **kwargs:
        These args are accepted:
            getCoords : bool.
                If getCoords, will return coords for each point.
            onlyCoords : bool.
                If true, with only return coords.
            verbose : bool, or int.
                If true or >1, will print evolution.

    Returns
    --------
    X : arr.
        The sample matrix. A nXd matrix, where n is the number of referenced pixels and d is the number of variables. Each line of the matrix is a pixel.
    Y : arr.
        The label of the pixel.

    See also
    ---------
    museotoolbox.vector_tools.readValuesFromVector : read field values from vector file.

    Examples
    ---------
    >>> from museotoolbox.datasets import getHistoricalMap
    >>> raster,vector=getHistoricalMap()
    >>> X,Y = getSamplesFromROI(raster,vector,'Class')
    >>> X
    array([[ 213.,  189.,  151.],
       [ 223.,  198.,  158.],
       [ 212.,  188.,  150.],
       ...,
       [ 144.,  140.,  105.],
       [  95.,   92.,   57.],
       [ 141.,  137.,  102.]])
    >>> X.shape
    (12647,3)
    >>> Y
    [3 3 3 ..., 1 1 1]
    >>> Y.shape
    (12647,)
    """
    # generate kwargs value
    if 'verbose' in kwargs.keys():
        verbose = kwargs['verbose']
    else:
        verbose = False
    if 'getCoords' in kwargs:
        getCoords = kwargs['getCoords']
    else:
        getCoords = False
    if 'onlyCoords' in kwargs:
        onlyCoords = kwargs['onlyCoords']
    else:
        onlyCoords = False
    # Open Raster
    raster = gdal.Open(inRaster, gdal.GA_ReadOnly)
    if raster is None:
        raise ImportError('Impossible to open ' + inRaster)
        # exit()
    # Convert vector to raster

    nFields = len(fields)
    if nFields == 0:
        fields = None
    rois = []
    temps = []
    for field in fields:
        rstField = tempfile.mktemp('_roi.tif')
        rstField = rasterize(inRaster, inVector, field,
                             rstField, gdal.GDT_Float64)
        roiField = gdal.Open(rstField, gdal.GA_ReadOnly)
        if roiField is None:
            raise Exception(
                'A problem occured when rasterizing {} with field {}'.format(
                    inVector, field))
        if (raster.RasterXSize != roiField.RasterXSize) or (
                raster.RasterYSize != roiField.RasterYSize):
            raise Exception('Raster and vector do not cover the same extent.')

        rois.append(roiField)
        temps.append(rstField)

    # Get block size
    band = raster.GetRasterBand(1)
    block_sizes = band.GetBlockSize()
    x_block_size = block_sizes[0]
    y_block_size = block_sizes[1]
    gdalDT = band.DataType
    del band

    # Get the number of variables and the size of the images
    d = raster.RasterCount
    nc = raster.RasterXSize
    nl = raster.RasterYSize

    # ulx, xres, xskew, uly, yskew, yres = raster.GetGeoTransform()

    if getCoords is True or onlyCoords is True:
        coords = np.array([], dtype=np.int64).reshape(0, 2)

    xDataType = convertGdalAndNumpyDataType(gdalDT)
    # Read block data
    X = np.array([], dtype=xDataType).reshape(0, d)
    F = np.array([], dtype=np.int64).reshape(
        0, nFields)  # now support multiple fields

    # for progress bar
    if verbose:
        total = 100
        pb = progressBar(total, message='Reading raster values... ')

    for i in range(0, nl, y_block_size):
        if i + y_block_size < nl:  # Check for size consistency in Y
            lines = y_block_size
        else:
            lines = nl - i
        for j in range(0, nc, x_block_size):  # Check for size consistency in X
            if j + x_block_size < nc:
                cols = x_block_size
            else:
                cols = nc - j

            # for progressbar
            if verbose:
                currentPosition = (i / nl) * 100
                pb.addPosition(currentPosition)
            # Load the reference data

            ROI = rois[0].GetRasterBand(1).ReadAsArray(j, i, cols, lines)

            t = np.nonzero(ROI)

            if t[0].size > 0:
                if getCoords or onlyCoords:
                    coordsTp = np.empty((t[0].shape[0], 2))

                    coordsTp[:, 0] = t[1] + j
                    coordsTp[:, 1] = t[0] + i

                    coords = np.concatenate((coords, coordsTp))

                # Load the Variables
                if not onlyCoords:
                    # extract values from each field
                    Ftemp = np.empty((t[0].shape[0], nFields), dtype=np.int64)
                    for idx, roiTemp in enumerate(rois):
                        roiField = roiTemp.GetRasterBand(
                            1).ReadAsArray(j, i, cols, lines)
                        Ftemp[:, idx] = roiField[t]
                    F = np.concatenate((F, Ftemp))

                    # extract raster values (X)
                    Xtp = np.empty((t[0].shape[0], d), dtype=xDataType)
                    for k in range(d):
                        band = raster.GetRasterBand(
                            k + 1).ReadAsArray(j, i, cols, lines)
                        Xtp[:, k] = band[t]
                    try:
                        X = np.concatenate((X, Xtp))
                    except MemoryError:
                        raise MemoryError(
                            'Impossible to allocate memory: ROI too big')

    if verbose:
        pb.addPosition(100)
    # Clean/Close variables
    # del Xtp,band
    roi = None  # Close the roi file
    raster = None  # Close the raster file

    # remove temp raster
    for roi in temps:
        os.remove(roi)

    # generate output
    if onlyCoords:
        toReturn = coords
    else:
        toReturn = [X] + [F[:, f] for f in range(nFields)]

        if getCoords:
            toReturn = toReturn + [coords]

    return toReturn


def rasterize(data, vectorSrc, field, outFile,
              gdt=gdal.GDT_Int16, invert=False):
    """
    Rasterize vector to the size of data (raster)

    Parameters
    -----------
    data : str
        path of raster
    vectorSrc : str
        path of vector
    field : str
        field to rasteirze (False is no field)
    outFile : str
        raster file where to save the rasterization
    gdt : int
        gdal GDT datatype (default gdal.GDT_Int16 = 3)
    invert : boolean.
        if invert, polygons will be masked.
    Returns
    --------
    outFile : str
    """

    dataSrc = gdal.Open(data)
    import ogr
    shp = ogr.Open(vectorSrc)

    lyr = shp.GetLayer()

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(
        outFile,
        dataSrc.RasterXSize,
        dataSrc.RasterYSize,
        1,
        gdt,
        options=['COMPRESS=DEFLATE'])
    dst_ds.SetGeoTransform(dataSrc.GetGeoTransform())
    dst_ds.SetProjection(dataSrc.GetProjection())

    if field is False or field is None:
        options = gdal.RasterizeOptions(inverse=invert)
        gdal.Rasterize(dst_ds, vectorSrc, options=options)
        dst_ds.GetRasterBand(1).SetNoDataValue(0)
    else:
        OPTIONS = ['ATTRIBUTE=' + field]
        gdal.RasterizeLayer(dst_ds, [1], lyr, None, options=OPTIONS)

    data, dst_ds, shp, lyr = None, None, None, None
    return outFile


class rasterMath:
    """
    Read a raster per block, and perform one or many functions to one or many raster outputs.
    If you want a sample of your data, just call getRandomBlock().

    Parameters
    ----------
    inRaster : str
        Gdal supported raster
    inMaskRaster : str or False.
        If str, path of the raster mask. Value masked are 0, other are considered not masked.

        Use `invert=True` in :mod:`museotoolbox.raster_tools.rasterMaskFromVector` to mask only what is not in polygons.
    return_3d : boolean, default False.
        Default will return a row per pixel (2 dimensions), and dimension 2 (bands) are columns.
        If return_3d is True, will return the block without reshape (not suitable to learn with `sklearn`).
    message : str or None.
        If str, the message will be displayed before the progress bar.

    Returns
    -------
    None

    Examples
    ---------
    >>> rM = rasterMath('tmp.tif')
    >>> rM.addFunction(np.mean,outRaster='/tmp/mean.tif',functionKwargs=dict(axis=1,dtype=np.int16))
    Using datatype from numpy table : int16
    >>> rM.run()
    Saved /tmp/mean.tif using function mean
    """

    def __init__(self, inRaster, inMaskRaster=False, return_3d=False,
                 message='rasterMath... ', verbose=True):

        self.verbose = verbose
        self.driver = gdal.GetDriverByName('GTiff')

        # Load raster
        self.openRaster = gdal.Open(inRaster, gdal.GA_ReadOnly)
        if self.openRaster is None:
            raise ReferenceError('Impossible to open ' + inRaster)

        self.nb = self.openRaster.RasterCount
        self.nc = self.openRaster.RasterXSize
        self.nl = self.openRaster.RasterYSize

        # Get the geoinformation
        self.GeoTransform = self.openRaster.GetGeoTransform()
        self.Projection = self.openRaster.GetProjection()

        # Get block size
        band = self.openRaster.GetRasterBand(1)
        self.block_sizes = band.GetBlockSize()
        self.x_block_size = self.block_sizes[0]
        self.y_block_size = self.block_sizes[1]
        self.nodata = band.GetNoDataValue()
        self.dtype = band.DataType
        self.ndtype = convertGdalAndNumpyDataType(band.DataType)
        self.return_3d = return_3d

        self.customBlockSize()  # get

        del band

        # Load inMask if given
        self.mask = inMaskRaster
        if self.mask:
            self.maskNoData = 0
            self.openMask = gdal.Open(inMaskRaster)

        # Initialize the output
        self.lastProgress = 0
        self.functions = []
        self.functionsKwargs = []
        self.outputs = []
        self.outputNoData = []

        # Initalize the run
        self.__position = 0

    def addFunction(
            self,
            function,
            outRaster,
            outNBand=False,
            outNumpyDT=False,
            outNoData=False,
            **kwargs):
        """
        Add function to rasterMath.

        Parameters
        ----------
        function : def.
            Function to parse with one arguments used to
        outRaster : str.
            Path of the raster to save the result.
        outNBand : int, default False.
            Number of bands of the outRaster.
            If False will take the number of dimensions from the first result of the function.
        outNumpyDT : int, default False.
            If False, will use the datatype of the function result.
        outNoData : int, default False.
            If False will use the minimum value available for the given or found datatype.
        functionKwargs : False, or dict type.
            If dict type, will be the other params of your function. E.g functionsKwargs = dict(axis=1).
        """

        if outNumpyDT is False:
            dtypeName = function(
                self.getRandomBlock().data,
                **kwargs).dtype.name
            outGdalDT = convertGdalAndNumpyDataType(numpyDT=dtypeName)
            pushFeedback('Using datatype from numpy table : ' + str(dtypeName))
        else:
            dtypeName = np.dtype(outNumpyDT).name
            outGdalDT = convertGdalAndNumpyDataType(numpyDT=dtypeName)

        if outNBand is False:
            randomBlock = function(self.getRandomBlock())
            if randomBlock.ndim > 1:
                outNBand = randomBlock.shape[randomBlock.ndim - 1]

            else:
                outNBand = 1
            if self.verbose:
                print('Detected {} band(s) for output.'.format(outNBand))

        self.__addOutput__(outRaster, outNBand, outGdalDT)
        self.functions.append(function)
        self.functionsKwargs.append(kwargs)

        if outNoData is False:
            if np.issubdtype(dtypeName, np.floating):
                outNoData = np.finfo(dtypeName).min
            else:
                outNoData = np.iinfo(dtypeName).min

        self.outputNoData.append(outNoData)

    def __addOutput__(self, outRaster, outNBand, outGdalDT):
        if not os.path.exists(os.path.dirname(outRaster)):
            os.makedirs(os.path.dirname(outRaster))
        dst_ds = self.driver.Create(
            outRaster, self.nc, self.nl, outNBand, outGdalDT, ['COMPRESS=DEFLATE'])
        dst_ds.SetGeoTransform(self.GeoTransform)
        dst_ds.SetProjection(self.Projection)

        self.outputs.append(dst_ds)

    def __iterBlock__(self, getBlock=False,
                      y_block_size=False, x_block_size=False):
        if not y_block_size:
            y_block_size = self.y_block_size
        if not x_block_size:
            x_block_size = self.x_block_size

        for row in range(0, self.nl, y_block_size):
            for col in range(0, self.nc, x_block_size):
                width = min(self.nc - col, x_block_size)
                height = min(self.nl - row, y_block_size)

                if getBlock:
                    X, mask = self.generateBlockArray(
                        col, row, width, height, self.mask)
                    yield X, mask, col, row, width, height
                else:
                    yield col, row, width, height

    def generateBlockArray(self, col, row, width, height, mask=False):
        """
        Return block according to position and width/height of the raster.

        Parameters
        ----------
        col : int.
            the col
        row : int
            the line.
        width : int.
            the width.
        height : int.
            the height;
        mask : bool.
            Use the mask.

        Returns
        -------
        arr : arr with values.
        arrMask : the masked array.
        """
        if self.return_3d:
            arr = np.empty((height, width, self.nb), dtype=self.ndtype)
        else:
            arr = np.empty((height * width, self.nb), dtype=self.ndtype)

        for ind in range(self.nb):
            band = self.openRaster.GetRasterBand(int(ind + 1))
            if self.return_3d:
                arr[:, :, ind] = band.ReadAsArray(
                    col, row, width, height)
            else:
                arr[:, ind] = band.ReadAsArray(
                    col, row, width, height).reshape(width * height)
        if mask:
            bandMask = self.openMask.GetRasterBand(1)
            arrMask = bandMask.ReadAsArray(
                col, row, width, height).astype(np.bool)
            if self.return_3d is False:
                arrMask = arrMask.reshape(width * height)
        else:
            arrMask = None

        arr, arrMask = self.filterNoData(arr, arrMask)

        return arr, arrMask

    def filterNoData(self, arr, mask=None):
        """
        Filter no data according to a mask and to nodata value set in the raster.
        """
        arrShape = arr.shape
        if self.return_3d:
            arrToCheck = np.copy(arr)[:, :, 0]
        else:
            arrToCheck = np.copy(arr)[:, 0]

        outArr = np.zeros((arrShape), dtype=self.ndtype)
        if self.nodata:
            outArr[:] = self.nodata

        if self.mask:
            t = np.logical_or((mask == False),
                              arrToCheck == self.nodata)
        else:
            t = np.where(arrToCheck == self.nodata)

        if self.return_3d:
            tmpMask = np.zeros(arrShape[:2], dtype=bool)
            tmpMask[t] = True
            tmpMask = np.repeat(tmpMask.reshape(*tmpMask.shape, 1), 3, axis=2)
            outArr = np.ma.masked_array(arr, tmpMask)
        else:
            tmpMask = np.zeros(arrShape, dtype=bool)
            tmpMask[t, :] = True
            outArr = np.ma.masked_array(arr, tmpMask)

        return outArr, tmpMask

    def getRandomBlock(self):
        """
        Get Random Block from the raster.
        """
        mask = np.array([True])

        while np.all(mask == True):
            cols = int(
                np.random.permutation(
                    range(
                        0,
                        self.nl,
                        self.y_block_size))[0])
            lines = int(
                np.random.permutation(
                    range(
                        0,
                        self.nc,
                        self.x_block_size))[0])
            width = min(self.nc - lines, self.x_block_size)
            height = min(self.nl - cols, self.y_block_size)

            tmp, _ = self.generateBlockArray(
                lines, cols, width, height, self.mask)
            mask = tmp.mask

        #arr = tmp[mask[:, 0], :]

        return tmp

    def readBlockPerBlock(self, x_block_size=False, y_block_size=False):
        """
        Yield each block.
        """
        for X, mask, col, line, cols, lines in self.__iterBlock__(
                getBlock=True, y_block_size=y_block_size, x_block_size=x_block_size):

            if not np.all(X.mask == 1):
                yield X

    def customBlockSize(self, x_block_size=False, y_block_size=False):
        """
        Define custom block size for reading/writing the raster.

        Parameters
        ----------
        y_block_size : int, default False.
            Integer, number of rows.
        x_block_size : int, default False.
            Integer, number of columns.
        """
        if y_block_size:
            if y_block_size == -1:
                self.y_block_size = self.nl
            elif isinstance(y_block_size, float):
                self.y_block_size = int(np.ceil(self.nl * y_block_size))
            else:
                self.y_block_size = y_block_size
        else:
            self.y_block_size = self.block_sizes[1]
        if x_block_size:
            if x_block_size == -1:
                self.x_block_size = self.nc
            elif isinstance(x_block_size, float):
                self.x_block_size = int(np.ceil(self.nc * x_block_size))
            else:
                self.x_block_size = x_block_size
        else:
            self.x_block_size = self.block_sizes[0]

        self.n_block = np.ceil(self.nl / self.y_block_size).astype(int) * np.ceil(self.nc /
                                                                                  self.x_block_size).astype(int)
        if self.verbose:
            print('Total number of blocks : %s' % self.n_block)
        self.pb = progressBar(self.n_block, message='rasterMath... ')

    def run(self, qgsFeedback=False):
        """
        Process with outside function.

        Parameters
        ----------
        verbose : bool or int.
            if >0
        Returns
        -------
        None
        """

        # TODO : Parallel/ Not working for now.

        for X, mask, col, line, cols, lines in self.__iterBlock__(
                getBlock=True):
            X_ = np.ma.copy(X)
            if qgsFeedback:
                if qgsFeedback == 'gui':
                    self.pb.addPosition(line)
                else:
                    qgsFeedback.setProgress(self.__position)
                    if qgsFeedback.isCanceled():
                        break

            if self.verbose:
                self.pb.addPosition(self.__position)

            for idx, fun in enumerate(self.functions):
                maxBands = self.outputs[idx].RasterCount
                if not np.all(X.mask == 1):
                    """
                    if self.return_3d:
                        mask = np.repeat(mask.reshape(*mask.shape,1),3,axis=2)
                        X_ = np.ma.masked_array(X,mask)
                    else:
                        X_ = X[~mask[:, 0], :]
                    """
                    if self.functionsKwargs[idx] is not False:
                        resFun = fun(X_, **
                                     self.functionsKwargs[idx])
                    else:
                        resFun = fun(X_)
                    if maxBands > self.nb:
                        if self.return_3d:
                            X = np.zeros((X_.shape[0], X_.shape[1], maxBands))
                            if self.outputNoData[idx] is not False:
                                X[:, :, :] = self.outputNoData[idx]
                            X[mask[:, :], :] = resFun
                        else:
                            """
                            X = np.zeros((X_.shape[0], maxBands))
                            if self.outputNoData[idx] is not False:
                                X[mask[:,0], :] = self.outputNoData[idx]
                            X[~mask[:, 0], :] = resFun
                            """
                    if resFun.ndim == 1:
                        resFun = resFun.reshape(-1, 1)
                    if resFun.ndim == 2 and self.return_3d is True:
                        nBands = 1
                    elif self.return_3d is True:
                        nBands = resFun.shape[2]
                    else:
                        nBands = resFun.shape[1]

                    if nBands > maxBands:
                        raise ValueError(
                            "Your function output {} bands, but has been defined to have a maximum of {} bands.".format(
                                resFun.shape[1], maxBands))
                    if not np.all(X.mask == 0):
                        if self.return_3d:
                            curX = np.where(
                                X_.mask[:, :, 0], self.outputNoData[idx], resFun)
                        else:
                            curX = np.where(np.repeat(
                                X_.mask[:, 0].reshape(-1, 1), nBands, axis=1), self.outputNoData[idx], resFun)
                    else:
                        curX = resFun
                else:
                    if self.outputNoData[idx] is not False:
                        if self.return_3d:
                            curX = np.full(
                                (X.shape[0], X.shape[1], maxBands), self.outputNoData[idx])
                        else:
                            curX = np.full(
                                (X.shape[0], maxBands), self.outputNoData[idx])
                    else:
                        raise ValueError(
                            'Some blocks are masked and no nodata value was given.\
                            \n Please give a nodata value when adding the function.')
                for ind in range(self.outputs[idx].RasterCount):
                    indGdal = int(ind + 1)
                    curBand = self.outputs[idx].GetRasterBand(indGdal)
                    if self.return_3d:
                        tmpX = np.copy(curX)
                        if curX.ndim == 2:
                            tmpX = curX.reshape(*curX.shape, 1)
                        tmpX = tmpX[:, :, ind]
                    else:
                        tmpX = curX[:, ind].reshape(lines, cols)

                    curBand.WriteArray(tmpX, col, line)
                    # curBand.FlushCache()

                band = self.outputs[idx].GetRasterBand(1)
                if self.outputNoData[idx] is not False:
                    band.SetNoDataValue(self.outputNoData[idx])
                band.FlushCache()

            self.__position += 1
        band = None
        self.pb.addPosition(self.n_block)

        for idx, fun in enumerate(self.functions):
            print(
                'Saved {} using function {}'.format(
                    self.outputs[idx].GetDescription(), str(
                        fun.__name__)))
            self.outputs[idx] = None
