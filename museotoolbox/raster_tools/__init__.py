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
The :mod:`museotoolbox.raster_tools` module gathers raster functions/classes.
"""
from osgeo import __version__ as osgeo_version
import gdal
import ogr
import numpy as np
import os
import tempfile

from ..internal_tools import progressBar, pushFeedback
from ..vector_tools import sampleExtraction

from scipy.ndimage.filters import generic_filter  # to compute moran


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
        "int8": 3,
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

        try:
            code = NP2GDAL_CONVERSION[numpyDT]
            if numpyDT.endswith('int64'):
                pushFeedback(
                    'Warning : Numpy type {} is not recognized by gdal. Will use int32 instead'.format(numpyDT))
        except BaseException:
            code = 7
            pushFeedback(
                'Warning : Numpy type {} is not recognized by gdal. Will use float64 instead'.format(numpyDT))
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

    if nFields == 0 or fields[0] == False:
        fields = [False]
    else:
        source = ogr.Open(inVector)
        layer = source.GetLayer()
        np_dtypes = []
        ldefn = layer.GetLayerDefn()
        for f in fields:
            idx = ldefn.GetFieldIndex(f)
            if idx == -1:
                raise ValueError('Field "{}" was not found.'.format(f))
            fdefn = ldefn.GetFieldDefn(idx)
            fdefn_type = fdefn.type
            if fdefn_type < 4 or fdefn_type == 12:
                if fdefn_type > 1 and fdefn_type != 12:
                    np_dtype = np.float64
                else:
                    np_dtype = np.int64
            else:
                raise ValueError(
                    'Wrong type for field "{}" : {}. \nPlease use int or float.'.format(
                        f, fdefn.GetFieldTypeName(
                            fdefn.type)))
            np_dtypes.append(np_dtype)

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
                    if nFields > 0:
                        Ftemp = np.empty(
                            (t[0].shape[0], nFields), dtype=np.int64)
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
                            'Impossible to allocate memory: ROI file is too big.')

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
        if nFields > 0:
            toReturn = [X] + [F[:, f] for f in range(nFields)]
        else:
            toReturn = X

        if getCoords:
            if nFields == 0:
                toReturn = [toReturn] + [coords]
            else:
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
        if invert == True:
            try:
                options = gdal.RasterizeOptions(inverse=invert)
                gdal.Rasterize(dst_ds, vectorSrc, options=options)
            except BaseException:
                raise Exception(
                    'Version of gdal is too old : RasterizeOptions is not available.\nPlease update.')
        else:
            #            gdal.Rasterize(dst_ds, vectorSrc)
            gdal.RasterizeLayer(dst_ds, [1], lyr, None)

        dst_ds.GetRasterBand(1).SetNoDataValue(0)
    else:
        OPTIONS = ['ATTRIBUTE=' + field]
        gdal.RasterizeLayer(dst_ds, [1], lyr, None, options=OPTIONS)

    data, dst_ds, shp, lyr = None, None, None, None
    return outFile


class rasterMath:
    """
    Read one or multiple rasters per block, and perform one or many functions to one or many raster outputs.
    If you want a sample of your data, just call getRandomBlock().

    The default option of rasterMath will return in 2d the dataset :
        - each line is a pixel with in columns its differents values in bands so masked data will not be given to this user.

    If you want to have the data in 3d (X,Y,Z), masked data will be given too (using numpy.ma).

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
    >>> import museotoolbox as mtb
    >>> raster,_=mtb.datasets.historicalMap()
    >>> rM = mtb.raster_tools.rasterMath(r)
    Total number of blocks : 15
    >>> rM.addFunction(np.mean,outRaster='/tmp/test.tif',axis=1,dtype=np.int16)
    Using datatype from numpy table : int16.
    Detected 1 band for function mean.
    >>> rM.run()
    rasterMath... [########################################]100%
    Saved /tmp/test.tif using function mean
    """

    def __init__(self, inRaster, inMaskRaster=False, return_3d=False,
                 message='rasterMath...', verbose=True):

        self.verbose = verbose
        self.message = message
        self.driver = gdal.GetDriverByName('GTiff')

        # Load raster
        self.openRasters = []

        self.addInputRaster(inRaster)

        self.nb = self.openRasters[0].RasterCount
        self.nc = self.openRasters[0].RasterXSize
        self.nl = self.openRasters[0].RasterYSize

        # Get the geoinformation
        self.GeoTransform = self.openRasters[0].GetGeoTransform()
        self.Projection = self.openRasters[0].GetProjection()

        # Get block size
        band = self.openRasters[0].GetRasterBand(1)
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
        self.options = [] # options is raster parameters

        # Initalize the run
        self.__position = 0

    def addInputRaster(
            self,
            inRaster):
        """
        Add input raster.

        Parameters
        -----------
        inRaster: str.
            Path of the raster.
        """
        openRaster = gdal.Open(inRaster, gdal.GA_ReadOnly)
        if openRaster is None:
            raise ReferenceError('Impossible to open image ' + inRaster)

        sameSize = True

        if len(self.openRasters) > 0:
            if openRaster.RasterXSize != self.openRasters[
                    0].RasterXSize or openRaster.RasterYSize != self.openRasters[0].RasterYSize:
                sameSize = False
                print("raster {} doesn't have the same size (X and Y) as the initial raster.\n \
                      Museotoolbox can't add it as an input raster.".format(os.path.basename(inRaster)))

        if sameSize:
            self.openRasters.append(openRaster)

    def addFunction(
            self,
            function,
            outRaster,
            outNBand=False,
            outNumpyDT=False,
            outNoData=False,
            compress=True,
            **kwargs):
        """
        Add function to rasterMath.

        Parameters
        ----------
        function : def.
            Function to parse with one arguments used to
        outRaster : str.
            Path of the raster to save the result.
        outNumpyDT : int, default False.
            If False, will use the datatype of the function result.
        outNoData : int, default True.
            If True or if False (but if nodata is present in the init raster),
            will use the minimum value available for the given or found datatype.
        compress: boolean, str ('high'),  default True.
            If True, will use DEFLATE compression using all cpu-cores minus 1.
            If 'high', will use DEFLATE with ZLEVEL = 9 and PREDICTOR=2.
        **kwargs

        """
        if len(kwargs) > 0:
            randomBlock = function(self.getRandomBlock(), **kwargs)
        else:
            randomBlock = function(self.getRandomBlock())
        if outNumpyDT is False:
            dtypeName = randomBlock.dtype.name
            outGdalDT = convertGdalAndNumpyDataType(numpyDT=dtypeName)
            pushFeedback(
                'Using datatype from numpy table : {}.'.format(dtypeName))
        else:
            dtypeName = np.dtype(outNumpyDT).name
            outGdalDT = convertGdalAndNumpyDataType(numpyDT=dtypeName)

        # get number of bands
        randomBlock = self.reshape_ndim(randomBlock)

        outNBand = randomBlock.shape[-1]
        need_s = ''
        if outNBand > 1:
            need_s = 's'

        if self.verbose:
            pushFeedback(
                'Detected {} band{} for function {}.'.format(
                    outNBand, need_s, function.__name__))
        
        if self.options == []:
            self.__initRasterParameters(compress=compress)
        
        self.__addOutput__(outRaster, outNBand, outGdalDT)
        self.functions.append(function)
        if len(kwargs) == 0:
            kwargs = False
        self.functionsKwargs.append(kwargs)

        if (outNoData is True) or (self.nodata is not None) or (
                self.mask is not False):
            if np.issubdtype(dtypeName, np.floating):
                minValue = np.finfo(dtypeName).min
            else:
                minValue = np.iinfo(dtypeName).min

            if not isinstance(outNoData, bool):
                if outNoData < minValue:
                    outNoData = minValue
            else:
                outNoData = minValue

            if self.verbose:
                pushFeedback('No data is set to : ' + str(outNoData))

        self.outputNoData.append(outNoData)
    
    def __initRasterParameters(self,compress=True):
        self.options = ['TILES=YES']
        self.options = []
        if compress is True or compress == 'high':
            n_jobs = os.cpu_count() - 1
            if n_jobs < 1 : n_jobs=1
            
            self.options.extend(['BIGTIFF=IF_SAFER','COMPRESS=DEFLATE'])
            
            if osgeo_version >= '2.1':
                self.options.append('NUM_THREADS={}'.format(n_jobs))

            if compress == 'high':
                self.options.append('PREDICTOR=2')
                self.options.append('ZLEVEL=9')
        else:
            self.options = ['BIGTIFF=IF_NEEDED']
        
    def getRasterParameters(self):
        """
        Get raster parameters (compression, block size...)
        
        Returns
        --------
        List of parameters
        
        References
        -----------
        As MuseoToolBox only saves in geotiff, parameters of gdal drivers for GeoTiff are here :
        https://gdal.org/drivers/raster/gtiff.html
        """
        if self.options == []:
            self.__initRasterParameters()
        return self.options
    
    def customRasterParameters(self,parameters_list):
        """
        Parameters to custom raster creation.
        
        Do not enter here blockXsize and blockYsize parameters as it is directly managed by :mod:`customBlockSize` function.
        
        Parameters
        -----------
        parameters_list : list.
            - example : ['BIGTIFF='IF_NEEDED','COMPRESS=DEFLATE']
            - example : ['COMPRESS=JPEG','JPEG_QUALITY=80']
        
        References
        -----------
        As MuseoToolBox only saves in geotiff, parameters of gdal drivers for GeoTiff are here :
        https://gdal.org/drivers/raster/gtiff.html
        """
        self.options = parameters_list 
        
    def __addOutput__(self, outRaster, outNBand, outGdalDT):
        if not os.path.exists(os.path.dirname(outRaster)):
            os.makedirs(os.path.dirname(outRaster))
        self.options.extend(['blockysize={}'.format(self.y_block_size),'blockxsize={}'.format(self.x_block_size)])            
        
        dst_ds = self.driver.Create(
            outRaster,
            self.nc,
            self.nl,
            outNBand,
            outGdalDT,
            options=self.options
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
        arrs = []
        if mask:
            bandMask = self.openMask.GetRasterBand(1)
            arrMask = bandMask.ReadAsArray(
                col, row, width, height).astype(np.bool)
            if self.return_3d is False:
                arrMask = arrMask.reshape(width * height)
        else:
            arrMask = None

        for nRaster in range(len(self.openRasters)):
            nb = self.openRasters[nRaster].RasterCount

            if self.return_3d:
                arr = np.empty((height, width, nb), dtype=self.ndtype)
            else:
                arr = np.empty((height * width, nb), dtype=self.ndtype)
            for ind in range(nb):
                band = self.openRasters[nRaster].GetRasterBand(int(ind + 1))

                if self.return_3d:
                    arr[..., ind] = band.ReadAsArray(
                        col, row, width, height)
                else:
                    arr[..., ind] = band.ReadAsArray(
                        col, row, width, height).reshape(width * height)

            arr = self.filterNoData(arr, arrMask)
            arrs.append(arr)

        if len(arrs) == 1:
            arrs = arrs[0]

        return arrs

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

    def getRandomBlock(self,seed=None):
        """
        Get Random Block from the raster.
        """
        mask = np.array([True])

        while np.all(mask == True):
            # TODO, stop and warn if no block has valid data (infinite loop...)
            np.random.seed(seed)
            cols = int(
                np.random.permutation(
                    range(
                        0,
                        self.nl,
                        self.y_block_size))[0])
            np.random.seed(seed)
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
            if len(self.openRasters) > 1:
                mask = tmp[0].mask
            else:
                mask = tmp.mask
            if self.return_3d is False:
                tmp = self._manageMaskFor2D(tmp)
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
        for nRaster in range(len(self.openRasters)):
            nb = self.openRasters[nRaster].RasterCount
            for n in range(1, nb + 1):
                band = self.openRasters[nRaster].GetRasterBand(n)
                band = band.ReadAsArray()
                if self.mask:
                    mask = np.asarray(
                        self.openMask.GetRasterBand(1).ReadAsArray(), dtype=bool)
                    band = np.ma.MaskedArray(band, mask=~mask)
                else:
                    band = np.ma.MaskedArray(
                        band, mask=np.where(
                            band == self.nodata, True, False))
                yield band

    def readBlockPerBlock(self, x_block_size=False, y_block_size=False):
        """
        Yield each block.
        """
        for X, col, line, cols, lines in self.__iterBlock__(
                getBlock=True, y_block_size=y_block_size, x_block_size=x_block_size):
            if isinstance(X, list):
                mask = X[0].mask
            else:
                mask = X.mask
            if not np.all(mask == 1):
                yield X

    def _returnUnmaskedX(self, X):
        if isinstance(X.mask, np.bool_):
            if X.mask == False:
                X = X.data
            else:
                pass
                # no return
        else:
            mask = np.in1d(X.mask[:, 0], True)
            X = X[~mask, :].data
        return X

    def _manageMaskFor2D(self, X):
        if len(self.openRasters) > 1:
            X = [self._returnUnmaskedX(x) for x in X]
        else:
            X = self._returnUnmaskedX(X)

        return X

    def customBlockSize(self, x_block_size=False, y_block_size=False):
        """
        Define custom block size for reading and writing the raster.

        Parameters
        ----------
        y_block_size : float or int, default False.
            IF integer, number of rows per block.
            If -1, means all the rows.
            If float, value must be between 0 and 1, such as 1/3.
        x_block_size : float or int, default False.
            If integer, number of columns per block.
            If -1, means all the columns.
            If float, value must be between 0 and 1, such as 1/3.
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
            pushFeedback('Total number of blocks : %s' % self.n_block)

    def run(self):
        """
        Process writing with outside function.

        Returns
        -------
        None
        """

        # TODO : Parallel
        self.pb = progressBar(self.n_block, message=self.message)

        for X, col, line, cols, lines in self.__iterBlock__(
                getBlock=True):

            if isinstance(X, list):
                X_ = [np.ma.copy(arr) for arr in X]
                X = X_[0]  # X_[0] is used to get mask
            else:
                X_ = np.ma.copy(X)

            if self.verbose:
                self.pb.addPosition(self.__position)

            for idx, fun in enumerate(self.functions):
                maxBands = self.outputs[idx].RasterCount

                if not np.all(X.mask == 1):
                    # if all the block is not masked
                    if not self.return_3d:
                        if isinstance(X_, list):
                            X__ = [arr[~X.mask[:, 0], ...].data for arr in X_]
                        else:
                            X__ = X[~X.mask[:, 0], ...].data
                    else:
                        X__ = np.ma.copy(X_)

                    if self.functionsKwargs[idx] is not False:
                        resFun = fun(X__, **
                                     self.functionsKwargs[idx])
                    else:
                        resFun = fun(X__)

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
                            tmp[mask.flatten(), ...] = self.outputNoData[idx]
                            tmp[~mask.flatten(), ...] = resFun
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

            if self.verbose:
                pushFeedback(
                    'Saved {} using function {}'.format(
                        self.outputs[idx].GetDescription(), str(
                            fun.__name__)))
            self.outputs[idx] = None


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
        self.lags = []
        if isinstance(inRaster, (np.ma.core.MaskedArray, np.ndarray)):
            arr = inRaster
        else:
            rM = rasterMath(
                inRaster,
                inMaskRaster=inMaskRaster,
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
                self.lags.append(l)
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
