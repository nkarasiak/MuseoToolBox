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
        "int64": 5,
        "uint64": 5
    }

    if numpyDT is None:
        code = gdal_array.GDALTypeCodeToNumericTypeCode(gdalDT)
    else:

        code = NP2GDAL_CONVERSION[numpyDT]
        if numpyDT.endswith('int64'):
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
    >>> from museotoolbox.datasets import historicalMap
    >>> from museotoolbox.raster_tools import getSamplesFromROI
    >>> raster,vector= historicalMap()
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
        options=['COMPRESS=DEFLATE', 'BIGTIFF=IF_SAFER'])
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
        Default will return a row per pixel (2 dimensions), and axis 2 (bands) are columns.
        If return_3d is True, will return the block without reshape (not suitable to learn with `sklearn`).
    message : str or None.
        If str, the message will be displayed before the progress bar.
    verbose : bool or integer.
        The higher is the integer verbose, the more it will returns informations.

    Examples
    ---------
    >>> rM = rasterMath('tmp.tif')
    >>> rM.addFunction(np.mean,outRaster='/tmp/mean.tif',functionKwargs=dict(axis=1,dtype=np.int16))
    Using datatype from numpy table : int16
    >>> rM.run()
    Saved /tmp/mean.tif using function mean
    """

    def __init__(self, inRaster, inMaskRaster=False, return_3d=False,
                 message='rasterMath...', verbose=True):

        self.verbose = verbose
        self.message = message
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
            outNoData=True,
            compress=False,
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
        outNoData : int, default True.
            If True or if False (but if nodata is present in the init raster),
            will use the minimum value available for the given or found datatype.
        compress: boolean, default False.
            If True, will use DEFLATE compression using all cpu-cores minus 1.
        **kwargs
            
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
            randomBlock = self.reshape_ndim(randomBlock)

            outNBand = randomBlock.shape[-1]
            if self.verbose:
                print(
                    'Detected {} band(s) for function {}.'.format(
                        outNBand, function.__name__))

        self.__addOutput__(outRaster, outNBand, outGdalDT,compress=compress)
        self.functions.append(function)
        self.functionsKwargs.append(kwargs)

        if (outNoData is True) or (self.nodata is not False):
            if np.issubdtype(dtypeName, np.floating):
                minValue = np.finfo(dtypeName).min
            else:
                minValue = np.iinfo(dtypeName).min
                
            if (outNoData is True) or (outNoData < minValue):
                outNoData = minValue
                
            if self.verbose:
                print('No data is set to : '+str(outNoData))
            
        self.outputNoData.append(outNoData)

    def __addOutput__(self, outRaster, outNBand, outGdalDT,compress=False):
        if not os.path.exists(os.path.dirname(outRaster)):
            os.makedirs(os.path.dirname(outRaster))
        if compress is True:
            options = ['BIGTIFF=IF_SAFER','COMPRESS=DEFLATE','NUM_THREADS={}'.format(os.cpu_count()-1)]
        else:
            options = ['BIGTIFF=IF_NEEDED']
        dst_ds = self.driver.Create(
            outRaster,
            self.nc,
            self.nl,
            outNBand,
            outGdalDT,
            options=options
            )
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
                    X = self.generateBlockArray(
                        col, row, width, height, self.mask)
                    yield X, col, row, width, height
                else:
                    yield col, row, width, height

    def generateBlockArray(self, col, row, width, height, mask=False):
        """
        Return block according to position and width/height of the raster.

        Parameters
        ----------
        col : int.
            the col.
        row : int
            the line.
        width : int.
            the width.
        height : int.
            the height.
        mask : bool.
            Use the mask (only if a mask if given in parameter of `rasterMath`.)

        Returns
        -------
        arr : numpy array with masked values. (`np.ma.masked_array`)
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

        arr = self.filterNoData(arr, arrMask)

        return arr

    def filterNoData(self, arr, mask=None):
        """
        Filter no data according to a mask and to nodata value set in the raster.
        """
        arrShape = arr.shape
        arrToCheck = np.copy(arr)[..., 0]

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
            tmpMask = np.repeat(tmpMask.reshape(*tmpMask.shape, 1), arr.shape[-1], axis=2)
            outArr = np.ma.masked_array(arr, tmpMask)
        else:
            tmpMask = np.zeros(arrShape, dtype=bool)
            tmpMask[t, :] = True
            outArr = np.ma.masked_array(arr, tmpMask)

        return outArr

    def getRandomBlock(self):
        """
        Get Random Block from the raster.
        """
        mask = np.array([True])

        while np.all(mask == True):
            # TODO, stop and warn if no block has valid data (infinite loop...)
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

            tmp = self.generateBlockArray(
                lines, cols, width, height, self.mask)
            mask = tmp.mask
        return tmp

    def reshape_ndim(self, x):
        """
        Reshape array with at least one band.

        Parameters
        ----------
        x : array.

        Returns
        -------
        x : array.

        """
        if x.ndim == 0:
            x = x.reshape(-1, 1)
        if self.return_3d:
            if x.ndim == 2:
                x = x.reshape(*x.shape, 1)
        else:
            if x.ndim == 1:
                x = x.reshape(-1, 1)
        return x

    def readBandPerBand(self):
        """
        Yields each whole band as np masked array (so with masked data)
        """
        for nb in range(1, self.nb + 1):
            band = self.openRaster.GetRasterBand(nb)
            band = band.ReadAsArray()
            if self.mask:
                mask = np.asarray(
                    self.openMask.GetRasterBand(1).ReadAsArray(), dtype=bool)
                band = np.ma.MaskedArray(band, mask=~mask)
            else:
                band = np.ma.MaskedArray(band)
            yield band

    def readBlockPerBlock(self, x_block_size=False, y_block_size=False):
        """
        Yield each block.
        """
        for X, col, line, cols, lines in self.__iterBlock__(
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
        self.pb = progressBar(self.n_block, message=self.message)

    def run(self):
        """
        Process with outside function.

        Parameters
        ----------

        Returns
        -------
        None
        """

        # TODO : Parallel

        for X, col, line, cols, lines in self.__iterBlock__(
                getBlock=True):
            X_ = np.ma.copy(X)

            if self.verbose:
                self.pb.addPosition(self.__position)

            for idx, fun in enumerate(self.functions):
                maxBands = self.outputs[idx].RasterCount

                if not np.all(X.mask == 1):
                    # if all the block is not masked
                    if not self.return_3d:
                        X_ = X_[~X_.mask[:,0],...]
                        
                    if self.functionsKwargs[idx] is not False:
                        resFun = fun(X_, **
                                     self.functionsKwargs[idx])
                    else:
                        resFun = fun(X_)

                    resFun = self.reshape_ndim(resFun)

                    nBands = resFun.shape[-1]
                    if nBands > maxBands:
                        raise ValueError(
                            "Your function output {} bands, but has been defined to have a maximum of {} bands.".format(
                                resFun.shape[1], maxBands))

                    if not np.all(X.mask == 0):
                        # if all the block is not unmasked add the nodata value
                                
                        resFun = self.reshape_ndim(resFun)
                        mask = self.reshape_ndim(X.mask[..., 0])
                        tmp = np.repeat(
                                mask,
                                maxBands,
                                axis=mask.ndim - 1)
                        if self.return_3d:
                            resFun = np.where(
                                    tmp,
                                    self.outputNoData[idx],
                                    resFun)                            
                        else:
                            tmp = tmp.astype(resFun.dtype)
                            tmp[mask.flatten(),...] = self.outputNoData[idx]
                            tmp[~mask.flatten(),...] = resFun
                            resFun = tmp
                            
                else:
                    # if all the block is masked
                    if self.outputNoData[idx] is not False:
                        # create an array with only the nodata value
                        # self.return3d+1 is just the right number of axis
                        resFun = np.full(
                            (*X.shape[:self.return_3d + 1], maxBands), self.outputNoData[idx])

                    else:
                        raise ValueError(
                            'Some blocks are masked and no nodata value was given.\
                            \n Please give a nodata value when adding the function.')

                for ind in range(maxBands):
                    # write result band per band
                    indGdal = ind + 1
                    curBand = self.outputs[idx].GetRasterBand(indGdal)

                    resToWrite = resFun[..., ind]

                    if self.return_3d is False:
                        # need to reshape as block
                        resToWrite = resToWrite.reshape(lines, cols)

                    curBand.WriteArray(resToWrite, col, line)
                    curBand.FlushCache()

            self.__position += 1

        self.pb.addPosition(self.n_block)

        for idx, fun in enumerate(self.functions):
            # set nodata if given
            if self.outputNoData[idx] is not False:
                band = self.outputs[idx].GetRasterBand(1)
                band.SetNoDataValue(self.outputNoData[idx])
                band.FlushCache()

            print(
                'Saved {} using function {}'.format(
                    self.outputs[idx].GetDescription(), str(
                        fun.__name__)))
            self.outputs[idx] = None
