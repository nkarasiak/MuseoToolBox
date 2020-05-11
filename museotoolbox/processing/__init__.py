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
The :mod:`museotoolbox.processing` module gathers raster and vector tools.
"""
# general libraries
import os
import numpy as np
import tempfile
from psutil import virtual_memory

# spatial libraries
from osgeo import __version__ as osgeo_version
from osgeo import gdal, ogr
from joblib import Parallel, delayed

from ..internal_tools import ProgressBar, push_feedback


def image_mask_from_vector(
        in_vector, in_image, out_image, invert=False, gdt=gdal.GDT_Byte):
    """
    Create a image mask where polygons/points are the pixels to keep.

    Parameters
    ----------
    in_vector : str.
        Path of the vector file to rasterize.
    in_image : str.
        Path of the raster file where the vector file will be rasterize.
    out_image : str.
        Path of the file (.tif) to create.
    invert : bool, optional (default=False).
        invert=True make polygons/points with 0 values in out_image.
    gdt : int, optional (default=gdal.GDT_Byte).
        The gdal datatype of the rasterized vector.
    """
    rasterize(
        in_image,
        in_vector,
        None,
        out_image,
        invert=invert,
        gdt=gdt)


def get_gdt_from_minmax_values(max_value, min_value=0):
    """
    Return the Gdal DataType according the minimum or the maximum value.

    Parameters
    ----------
    max_value : int or float.
        The maximum value needed.
    min_value : int or float, optional (default=0).
        The minimum value needed.

    Returns
    -------
    gdalDT : int.
        gdal datatype.

    Examples
    ---------
    >>> get_gdt_from_minmax_values(260)
    2
    >>> get_gdt_from_minmax_values(16)
    1
    >>> get_gdt_from_minmax_values(16,-260)
    3
    """
    max_abs_value = np.amax(np.abs([max_value, min_value]))

    # if values are int
    if isinstance(max_abs_value, (int, np.integer)):
        if min_value >= 0:
            if max_value <= 255:
                gdalDT = gdal.GDT_Byte
            elif max_value > 255 and max_value <= 65535:
                gdalDT = gdal.GDT_UInt16
            elif max_value >= 65535:
                gdalDT = gdal.GDT_UInt32
        elif min_value < 0:
            if min_value > -65535:
                gdalDT = gdal.GDT_Int16
            else:
                gdalDT = gdal.GDT_Int32

    # if values are float
    if isinstance(max_abs_value, float):
        if max_abs_value > +3.4E+38:
            gdalDT = gdal.GDT_Float64
        else:
            gdalDT = gdal.GDT_Float32

    return gdalDT


def convert_dt(dt, to_otb_dt=False):
    """
    Return the datatype from gdal to numpy or from numpy to gdal.

    Parameters
    -----------
    dt : int or str
        gdal datatype from src_dataset.GetRasterBand(1).DataType.
        numpy datatype from np.array([]).dtype.name

    Returns
    --------
        dt : int or data type
            - For gdal, the data type (int).
            - For numpy, the date type (type).

    Examples
    ---------
    >>> _convert_dt(gdal.GDT_Int16)
    numpy.int16
    >>> _convert_dt(gdal.GDT_Float64)
    numpy.float64
    >>> _convert_dt(numpyDT=np.array([],dtype=np.int16).dtype.name)
    3
    >>> _convert_dt(numpyDT=np.array([],dtype=np.float64).dtype.name)
    7
    """
    from osgeo import gdal_array
    if isinstance(dt, int):
        is_gdal = True
    else:
        is_gdal = False

    if is_gdal is True:
        code = gdal_array.GDALTypeCodeToNumericTypeCode(dt)
    else:
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
        try:
            code = NP2GDAL_CONVERSION[dt]
            if dt.endswith('int64'):
                push_feedback(
                    'Warning : Numpy type {} is not recognized by gdal. Will use int32 instead'.format(dt))
        except BaseException:
            code = 7
            push_feedback(
                'Warning : Numpy type {} is not recognized by gdal. Will use float64 instead'.format(dt))
    if to_otb_dt:
        if is_gdal:
            code = _convert_gdal_to_otb_dt(dt)
        else:
            code = _convert_gdal_to_otb_dt(code)
    return code


def _convert_gdal_to_otb_dt(dt):
    """
    Convert Gdal DataType to OTB str format.

    Parameters
    ----------
    dt : int
        gdal datatype from src_dataset.GetRasterBand(1).DataType.

    Returns
    ----------
    otb_dt : str.
        The otb data type.

    Examples
    ---------
    >>> _convert_gdal_to_otb_dt(gdal.GDT_Float32)
    'float'
    >>> _convert_gdal_to_otb_dt(gdal.GDT_Byte)
    'uint8'
    >>> _convert_gdal_to_otb_dt(gdal.GDT_UInt32)
    'uint32'
    >>> _convert_gdal_to_otb_dt(gdal.GDT_CFloat64)
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

    if dt > len(code):
        otb_dt = ('cdouble')
    else:
        otb_dt = code[dt]

    return otb_dt


def extract_ROI(in_image, in_vector, *fields, **kwargs):
    """
    Extract raster values from Regions Of Interest in a vector file.
    Initially written by Mathieu Fauvel, improved by Nicolas Karasiak.

    Parameters
    -----------
    in_image : str.
        the name or path of the raster file, could be any file that GDAL can open.
    in_vector : str.
        A filename or path corresponding to a vector file.
        It could be any file that GDAL/OGR can open.
    *fields : str.
        Each field to extract label/value from.
    **kwargs : list of kwargs.
        - get_pixel_position : bool, optional (default=False).
            If `get_pixel_position=True`, will return pixel position in the image for each point.
        - only_pixel_position : bool, optional (default=False).
            If `only_pixel_position=True`, with only return pixel position for each point.
        - prefer_memory : bool, optional (default=False).
            If `prefer_memory=False`, will write temporary raster on disk to extract ROI values.
        - verbose : bool or int, optional (default=True).
            The higher is the int verbose, the more it will returns informations.

    Returns
    --------
    X : np.ndarray, size of (n_samples,n_features).
        The sample matrix.
        A n*d matrix, where n is the number of referenced pixels and d is the number of features.
        Each line of the matrix is a pixel.
    y : np.ndarray, size of (n_samples,).
        The label of each pixel.

    See also
    ---------
    museotoolbox.processing.read_vector_values : read field values from vector file.

    Examples
    ---------
    >>> from museotoolbox.datasets import load_historical_data
    >>> from museotoolbox.processing import extract_ROI
    >>> raster,vector= load_historical_data()
    >>> X,Y = extract_ROI(raster,vector,'Class')
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
    if 'get_pixel_position' in kwargs:
        get_pixel_position = kwargs['get_pixel_position']
    else:
        get_pixel_position = False
    if 'only_pixel_position' in kwargs:
        only_pixel_position = kwargs['only_pixel_position']
    else:
        only_pixel_position = False
    if 'prefer_memory' in kwargs:
        prefer_memory = kwargs['prefer_memory']
    else:
        prefer_memory = False
    # Open Raster
    raster = gdal.Open(in_image, gdal.GA_ReadOnly)
    if raster is None:
        raise ValueError('Impossible to open ' + in_image)
        # exit()
    # Convert vector to raster

    nFields = len(fields)

    if nFields == 0 or fields[0] == False:
        fields = [False]
    else:
        source = ogr.Open(in_vector)
        layer = source.GetLayer()
        np_dtypes = []
        ldefn = layer.GetLayerDefn()
        for f in fields:
            idx = ldefn.GetFieldIndex(f)
            if idx == -1:

                listFields = []
                for n in range(ldefn.GetFieldCount()):
                    fdefn = ldefn.GetFieldDefn(n)
                    if fdefn.name is not listFields:
                        listFields.append('"' + fdefn.name + '"')
                raise ValueError(
                    'Sorry, field "{}" is not available.\nThese fields are available : {}.'.format(
                        f, ', '.join(listFields)))

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
        if prefer_memory:
            raster_in_mem = True
            image_field = 'MEM'
            data_src = rasterize(in_image, in_vector, field,
                                 out_image=image_field, gdt=gdal.GDT_Float64)

        else:

            raster_in_mem = False
            image_field = tempfile.mktemp('_roi.tif')
            rasterize(in_image, in_vector, field,
                      out_image=image_field, gdt=gdal.GDT_Float64)
            data_src = gdal.Open(image_field, gdal.GA_ReadOnly)
            temps.append(image_field)

        if data_src is None:
            raise Exception(
                'A problem occured when rasterizing {} with field {}'.format(
                    in_vector, field))
        if (raster.RasterXSize != data_src.RasterXSize) or (
                raster.RasterYSize != data_src.RasterYSize):
            raise Exception('Raster and vector do not cover the same extent.')

        rois.append(data_src)

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

    if get_pixel_position is True or only_pixel_position is True:
        coords = np.array([], dtype=np.int64).reshape(0, 2)

    xDataType = convert_dt(gdalDT)

    # Read block data
    X = np.array([], dtype=xDataType).reshape(0, d)
    F = np.array([], dtype=np.int64).reshape(
        0, nFields)  # now support multiple fields

    # for progress bar
    if verbose:
        total = 100
        pb = ProgressBar(total, message='Reading raster values... ')

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

            # for ProgressBar
            if verbose:
                currentPosition = (i / nl) * 100
                pb.add_position(currentPosition)
            # Load the reference data

            ROI = rois[0].GetRasterBand(1).ReadAsArray(j, i, cols, lines)

            t = np.nonzero(ROI)

            if t[0].size > 0:
                if get_pixel_position or only_pixel_position:
                    coordsTp = np.empty((t[0].shape[0], 2))

                    coordsTp[:, 0] = t[1] + j
                    coordsTp[:, 1] = t[0] + i

                    coords = np.concatenate((coords, coordsTp))

                # Load the Variables
                if not only_pixel_position:
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

                    X = np.concatenate((X, Xtp))

    if verbose:
        pb.add_position(100)
    # Clean/Close variables
    # del Xtp,band
    roi = None  # Close the roi file
    raster = None  # Close the raster file

    # remove temp raster
    if raster_in_mem:
        for roi in temps:
            os.remove(roi)

    # generate returns
    if only_pixel_position:
        toReturn = coords
    else:
        if nFields > 0:
            toReturn = [X] + [F[:, f] for f in range(nFields)]
        else:
            toReturn = X

        if get_pixel_position:
            if nFields == 0:
                toReturn = [toReturn] + [coords]
            else:
                toReturn = toReturn + [coords]

    return toReturn


def rasterize(in_image, in_vector, in_field=False, out_image='MEM',
              gdt=gdal.GDT_Int16, invert=False):
    """
    Rasterize vector to the size of data (raster)

    Parameters
    -----------
    in_image : str.
        A filename or path corresponding to a raster image.
    in_vector : str.
        A filename or path corresponding to a vector file.
    in_field : str, optional (default=False).
        Name of the filed to rasteirze.
        If False, will rasterize the polygons or points with >0 value, and set the other values to 0.
    out_image : str, optional (default = 'MEM').
        A filename or path corresponding to a geotiff (.tif) raster image to save.
        'MEM' will store raster in memory.
    gdt : int, optional (default=gdal.GDT_Int16)
        gdal datatype.
    invert : bool, optional (default=False).
        if invert is True, polygons will have 0 values in the out_image.

    Returns
    --------
     dst_ds : gdal object
         The open dataset with gdal (essential if out_image is set to 'MEM')
    """

    data_src = gdal.Open(in_image)
    shp = ogr.Open(in_vector)

    lyr = shp.GetLayer()

    if out_image.upper() == 'MEM':

        driver = gdal.GetDriverByName('MEM')
        out_image = ''
        options = []
    else:

        driver = gdal.GetDriverByName('GTiff')
        options = ['COMPRESS=PACKBITS', 'BIGTIFF=IF_SAFER']

    dst_ds = driver.Create(
        out_image,
        data_src.RasterXSize,
        data_src.RasterYSize,
        1,
        gdt,
        options=options)
    dst_ds.SetGeoTransform(data_src.GetGeoTransform())
    dst_ds.SetProjection(data_src.GetProjection())

    if in_field is False or in_field is None:
        if invert:
            try:
                options = gdal.RasterizeOptions(inverse=invert)
                gdal.Rasterize(dst_ds, in_vector, options=options)
            except BaseException:
                raise Exception(
                    'Version of gdal is too old : RasterizeOptions is not available.\nPlease update.')
        else:
            #            gdal.Rasterize(dst_ds, vectorSrc)
            gdal.RasterizeLayer(dst_ds, [1], lyr, options=options)

        dst_ds.GetRasterBand(1).SetNoDataValue(0)
    else:
        options = ['ATTRIBUTE=' + in_field]
        gdal.RasterizeLayer(dst_ds, [1], lyr, None, options=options)

    data_src, shp, lyr = None, None, None

    return dst_ds


class RasterMath:
    """
    Read one or multiple rasters per block, and perform one or many functions to one or many geotiff raster outputs.
    If you want a sample of your data, just call :func:`~museotoolbox.processing.RasterMath.get_random_block`.

    The default option of rasterMath will return in 2d the dataset :
        - each line is a pixel with in columns its differents values in bands so masked data will not be given to this user.

    If you want to have the data in 3d (X,Y,Z), masked data will be given too (using numpy.ma).

    Parameters
    ----------
    in_image : str.
        Path of a gdal extension supported raster.
    in_image_mask : str or False, optional (default=False).
        If str, path of the raster mask. Value masked are 0, other are considered not masked.
        Use ``invert=True`` in :mod:`museotoolbox.processing.image_mask_from_vector` to mask only what is not in polygons.
    return_3d : bool, optional (default=False).
        Default will return a row per pixel (2 dimensions), and axis 2 (bands) are columns.
        If ``return_3d=True``, will return the block without reshape (not suitable to learn with `sklearn`).
    block_size : list or False, optional (default=[256,256]).
        Define the reading and writing block size. First element is the number of columns, second element the number of lines per block.
        If False, will use the block size as defined in in_image.
        To define later the block_size, use `custom_block_size`.
    n_jobs : int, optional, (default value : 1)
        Numbers of workers or process that will work in parallel.
        To use if your function are very time consumming.
    message : str, optional (default='rasterMath...').
        If str, the message will be displayed before the progress bar.
    verbose : bool or int, optional (default=True).
        The higher is the int verbose, the more it will returns informations.

    Examples
    ---------
    >>> import museotoolbox as mtb
    >>> raster,_= mtb.datasets.load_historical_data()
    >>> rM = mtb.processing.RasterMath(raster)
    Total number of blocks : 15
    >>> rM.add_function(np.mean,out_image='/tmp/test.tif',axis=1,dtype=np.int16)
    Using datatype from numpy table : int16.
    Detected 1 band for function mean.
    >>> rM.run()
    rasterMath... [########################################]100%
    Saved /tmp/test.tif using function mean
    """

    def __init__(
            self,
            in_image,
            in_image_mask=False,
            return_3d=False,
            block_size=[
                256,
                256],
            n_jobs=1,
            message='rasterMath...',
            verbose=True):

        self.verbose = verbose
        self.message = message
        self.driver = gdal.GetDriverByName('GTiff')
        self.itemsize = 0

        # Load raster
        self.opened_images = []

        self.add_image(in_image)

        if n_jobs < 0:
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs
        self.n_bands = self.opened_images[0].RasterCount
        self.n_columns = self.opened_images[0].RasterXSize
        self.n_lines = self.opened_images[0].RasterYSize

        # Get the geoinformation
        self.geo_transform = self.opened_images[0].GetGeoTransform()
        self.projection = self.opened_images[0].GetProjection()

        # Get block size and itemsize
        band = self.opened_images[0].GetRasterBand(1)
        self.input_block_sizes = band.GetBlockSize()

        # input block size
        if block_size is False:
            self.x_block_size = self.input_block_sizes[0]
            self.y_block_size = self.input_block_sizes[1]
        else:
            self.x_block_size = block_size[0]
            self.y_block_size = block_size[1]
        self.block_sizes = [self.x_block_size, self.y_block_size]
        self.custom_block_size()  # set block size

        self.nodata = band.GetNoDataValue()
        self.dtype = band.DataType
        self.dtype_np = convert_dt(band.DataType)
        self.return_3d = return_3d

        del band  # for memory purposes

        # Load in_image_mask if given
        self.mask = in_image_mask
        if self.mask:
            self.opened_mask = gdal.Open(in_image_mask)
            if self.opened_mask is None:
                raise ReferenceError(
                    'Impossible to open image ' + in_image_mask)

        # Initialize the output
        self.lastProgress = 0

        self._outputs = []
        self._raster_options = {}

    def add_image(
            self,
            in_image):
        """
        Add raster image.

        Parameters
        -----------
        in_image : str.
            Path of a gdal extension supported raster.
        """

        opened_raster = gdal.Open(in_image, gdal.GA_ReadOnly)
        if opened_raster is None:
            raise ReferenceError('Impossible to open image ' + in_image)

        sameSize = True

        if len(self.opened_images) > 0:
            if opened_raster.RasterXSize != self.opened_images[
                    0].RasterXSize or opened_raster.RasterYSize != self.opened_images[0].RasterYSize:
                sameSize = False
                raise ValueError(
                    "raster {} doesn't have the same size (X and Y) as the initial raster.\n \
                      Museotoolbox can't add it as an input raster.".format(
                        os.path.basename(in_image)))
        n_bands = opened_raster.RasterCount

        band = opened_raster.GetRasterBand(1)
        mem_size = band.ReadAsArray(0, 0, 1, 1).itemsize * n_bands
        self.itemsize += mem_size
        del band
#        self.itemsize += opened_raster.GetRasterBand(0).ReadAsArray(0,0,1,1).itemsize*n_bands

        if sameSize:
            self.opened_images.append(opened_raster)

    def add_function(
            self,
            function,
            out_image,
            out_n_bands=False,
            out_np_dt=False,
            out_nodata=False,
            compress=True,
            **kwargs):
        """
        Add function to rasterMath.

        Parameters
        ----------
        function : function.
            Function to parse where the first argument is a numpy array similar to what :mod:`museotoolbox.processing.RasterMath.get_random_block()` returns.
        out_image : str.
            A path to a geotiff extension filename corresponding to a raster image to create.
        out_n_bands : int or False, optional (default=False).
            If False, will run the given function to find the number of bands to define in the out_image.
        out_np_dt : int or False, optional (default=False).
            If False, will run the given function to get the datatype.
        out_nodata : int, float or False, optional (default=False).
            If True or if False (if nodata is present in the init raster),
            will use the minimum value available for the given or found datatype.
        compress: bool or str, optional (default=True).
            If True, will use PACKBITS.
            If 'high', will use DEFLATE with ZLEVEL = 9 and PREDICTOR=2.
        **kwargs :
            kwargs are keyword arguments you need for the given function.

        See also
        ----------
        museotoolbox.processing.RasterMath.get_random_block : To test your function, parse the first argument with a random block
        museotoolbox.processing.convert_dt : To see conversion between numpy datatype to gdal datatype.
        museotoolbox.processing.get_dt_from_minmax_values : To get the gdal datatype according to a min/max value.
        """
        if len(kwargs) > 0:
            randomBlock = function(self.get_random_block(), **kwargs)
        else:
            randomBlock = function(self.get_random_block())
        if out_np_dt is False:
            np_type = randomBlock.dtype
            dtypeName = np_type.name
            out_np_dt = convert_dt(dtypeName)
            if self.verbose:
                push_feedback(
                    'Using datatype from numpy table : {}.'.format(dtypeName))
        else:
            dtypeName = np.dtype(out_np_dt).name
            out_np_dt = convert_dt(dtypeName)

        # get number of bands
        randomBlock = self.reshape_ndim(randomBlock)

        out_n_bands = randomBlock.shape[-1]
        need_s = ''
        if out_n_bands > 1:
            need_s = 's'

        if self.verbose:
            push_feedback(
                'Detected {} band{} for function {}.'.format(
                    out_n_bands, need_s, function.__name__))

        if self._raster_options == []:
            self._init_raster_parameters(compress=compress)
        else:
            params = self.get_raster_parameters()
            arg_pos = next(
                (x for x in params if x.startswith('compress')), None)
            if arg_pos:
                # remove old compress arg
                params.pop(params.index(arg_pos))
            self.custom_raster_parameters(params)
            self._init_raster_parameters(compress=compress)

        self._add_output(out_image, out_n_bands, out_np_dt)

        if len(kwargs) == 0:
            kwargs = False

        if (out_nodata is True) or (self.nodata is not None) or (
                self.mask is not False):
            if np.issubdtype(dtypeName, np.floating):
                minValue = float(np.finfo(dtypeName).min)
            else:
                minValue = np.iinfo(dtypeName).min

            if not isinstance(out_nodata, bool):
                if out_nodata < minValue:
                    out_nodata = minValue
            else:
                out_nodata = minValue

            if self.verbose:
                push_feedback('No data is set to : {}.'.format(out_nodata))

        self._outputs[-1]['gdal_type'] = out_np_dt
        self._outputs[-1]['np_type'] = dtypeName
        self._outputs[-1]['function'] = function
        self._outputs[-1]['n_bands'] = out_n_bands
        self._outputs[-1]['kwargs'] = kwargs
        self._outputs[-1]['nodata'] = out_nodata
        self._outputs[-1]['itemsize'] = randomBlock.itemsize * out_n_bands

    def _init_raster_parameters(self, compress=True):

        self._raster_options = []

        if compress:

            self._raster_options.append('BIGTIFF=IF_SAFER')

            if osgeo_version >= '2.1':
                self._raster_options.append(
                    'NUM_THREADS={}'.format(self.n_jobs))

            if compress == 'high':

                self._raster_options.append('COMPRESS=DEFLATE')
                self._raster_options.append('PREDICTOR=2')
                self._raster_options.append('ZLEVEL=9')
            else:
                self._raster_options.append('COMPRESS=PACKBITS')
        else:
            self._raster_options = ['BIGTIFF=IF_NEEDED']

    def get_raster_parameters(self):
        """
        Get raster parameters (compression, block size...)

        Returns
        --------
        options : list.
            List of parameters for creating the geotiff raster.

        References
        -----------
        As MuseoToolBox only saves in geotiff, parameters of gdal drivers for GeoTiff are here :
        https://gdal.org/drivers/raster/gtiff.html
        """
        if self._raster_options == []:
            self._init_raster_parameters()
        return self._raster_options

    def custom_raster_parameters(self, parameters_list):
        """
        Parameters to custom raster creation.

        Do not enter here blockXsize and blockYsize parameters as it is directly managed by :mod:`custom_block_size` function.

        Parameters
        -----------
        parameters_list : list.
            - example : ['BIGTIFF=IF_NEEDED','COMPRESS=DEFLATE']
            - example : ['COMPRESS=JPEG','JPEG_QUALITY=80']

        References
        -----------
        As MuseoToolBox only saves in geotiff, parameters of gdal drivers for GeoTiff are here :
        https://gdal.org/drivers/raster/gtiff.html

        See also
        ---------
        museotoolbox.processing.RasterMath.custom_block_size : To custom the reading and writing block size.
        """
        self._raster_options = parameters_list

    def _managed_raster_parameters(self):
        # remove blockysize or blockxsize if already in options
        self._raster_options = [val for val in self._raster_options if not val.upper().startswith(
            'BLOCKYSIZE') and not val.upper().startswith('BLOCKXSIZE') and not val.upper().startswith('TILED')]
        self._raster_options.extend(['BLOCKYSIZE={}'.format(
            self.y_block_size), 'BLOCKXSIZE={}'.format(self.x_block_size)])
        if self.y_block_size == self.x_block_size:
            if self.y_block_size in [64, 128, 256, 512, 1024, 2048, 4096]:
                self._raster_options.extend(['TILED=YES'])

    def _add_output(self, out_image, out_n_bands, out_np_dt):
        if not os.path.exists(os.path.dirname(os.path.abspath(out_image))):
            os.makedirs(os.path.dirname(os.path.abspath(out_image)))

        self._managed_raster_parameters()

        dst_ds = self.driver.Create(
            out_image,
            self.n_columns,
            self.n_lines,
            out_n_bands,
            out_np_dt,
            options=self._raster_options
        )
        dst_ds.SetGeoTransform(self.geo_transform)
        dst_ds.SetProjection(self.projection)

        self._outputs.append(dict(gdal_object=dst_ds))

    def _iter_block(self, get_block=False,
                    y_block_size=False, x_block_size=False):
        if not y_block_size:
            y_block_size = self.y_block_size
        if not x_block_size:
            x_block_size = self.x_block_size

        for row in range(0, self.n_lines, y_block_size):
            for col in range(0, self.n_columns, x_block_size):
                width = min(self.n_columns - col, x_block_size)
                height = min(self.n_lines - row, y_block_size)

                if get_block:
                    X = self._generate_block_array(
                        col, row, width, height, self.mask)
                    yield X, col, row, width, height
                else:
                    yield col, row, width, height

    def _generate_block_array(self, col, row, width, height, mask=False):
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
            Use the mask (only if a mask if given in parameter of `RasterMath`.)

        Returns
        -------
        arr : numpy array with masked values. (`np.ma.masked_array`)
        """
        arrs = []
        if mask:
            bandMask = self.opened_mask.GetRasterBand(1)
            arrMask = bandMask.ReadAsArray(
                col, row, width, height).astype(np.bool)
            if self.return_3d is False:
                arrMask = arrMask.reshape(width * height)
        else:
            arrMask = None

        for nRaster in range(len(self.opened_images)):
            nb = self.opened_images[nRaster].RasterCount

            if self.return_3d:
                arr = np.empty((height, width, nb), dtype=self.dtype_np)
            else:
                arr = np.empty((height * width, nb), dtype=self.dtype_np)
#            for ind in range(nb):
            arr = self.opened_images[nRaster].ReadAsArray(
                col, row, width, height)
            if arr.ndim > 2:
                arr = np.moveaxis(arr, 0, -1)

            if not self.return_3d:
                if arr.ndim == 2:
                    arr = arr.flatten()
                else:
                    arr = arr.reshape(-1, arr.shape[-1])

            arr = self._filter_nodata(arr, arrMask)
            arrs.append(arr)

        if len(arrs) == 1:
            arrs = arrs[0]

        return arrs

    def _filter_nodata(self, arr, mask=None):
        """
        Filter no data according to a mask and to nodata value set in the raster.
        """
        arrShape = arr.shape
        arrToCheck = np.copy(arr)[..., 0]

        outArr = np.zeros((arrShape), dtype=self.dtype_np)
        if self.nodata:
            outArr[:] = self.nodata

        if self.mask:
            t = np.logical_or((mask == False),
                              arrToCheck == self.nodata)
        else:
            t = np.where(arrToCheck == self.nodata)

        if self.return_3d:
            if arr.ndim == 2:
                arr = np.expand_dims(arr, 2)
            tmpMask = np.zeros(arrShape[:2], dtype=bool)
            tmpMask[t] = True
            tmpMask = np.repeat(tmpMask.reshape(
                *tmpMask.shape, 1), arr.shape[-1], axis=2)
            outArr = np.ma.masked_array(arr, tmpMask)
        else:
            tmpMask = np.zeros(arrShape, dtype=bool)
            tmpMask[t, ...] = True
            outArr = np.ma.masked_array(arr, tmpMask)

        return outArr

    def get_block(self, block_number=0, return_with_mask=False):
        """
        Get a block by its position, ordered as follow :

        +-----------+-----------+
        |  block 0  |  block 1  |
        +-----------+-----------+
        |  block 2  |  block 3  |
        +-----------+-----------+

        Parameters
        -----------
        block_number : int, optional (default=0).
            Position of the desired block.
        return_with_mask : bool, optinal (default=False)
            If True and if return_3d is True, will return a numpy masked array.

        Returns
        --------
        Block : np.ndarray or np.ma.MaskedArray
        """

        if block_number > self.n_blocks:
            raise ValueError(
                'There are only {} blocks in your image.'.format(
                    self.n_blocks))
        else:

            row = [l for l in range(0, self.n_lines, self.y_block_size)]
            col = [c for c in range(0, self.n_columns, self.x_block_size)]

            row_number = int(block_number / self.n_x_blocks)
            col_number = int(block_number % self.n_x_blocks)

            width = min(self.n_columns - col[col_number], self.x_block_size)
            height = min(self.n_lines - row[row_number], self.y_block_size)

            tmp = self._generate_block_array(
                col[col_number], row[row_number], width, height, self.mask)

            # return only available pixels when user ask and it is 2d
            if return_with_mask is False and self.return_3d is False:
                if len(self.opened_images) > 1:
                    tmp = [np.ma.copy(t) for t in tmp]
                    tmp = [t.data for t in tmp]
                else:
                    tmp = np.ma.copy(tmp)
                    tmp = tmp.data

            return tmp

    def get_block_coords(self, block_number=0):
        """
        Get position of a block :

            order as [x,y,width,height]

        Parameters
        -----------
        block_number, int, optional (default=0).
            Position of the desired block.

        Returns
        --------
        List of positions of the block [x,y,width,height]

        """
        if block_number > self.n_blocks:
            raise ValueError(
                'There are only {} blocks in your image.'.format(
                    self.n_blocks))
        else:

            row = [l for l in range(0, self.n_lines, self.y_block_size)]
            col = [c for c in range(0, self.n_columns, self.x_block_size)]

            row_number = int(block_number / self.n_x_blocks)
            col_number = int(block_number % self.n_x_blocks)

            width = min(self.n_columns - col[col_number], self.x_block_size)
            height = min(self.n_lines - row[row_number], self.y_block_size)

            return [col[col_number], row[row_number], width, height]

    def _manage_block_mask(self, block):

        if isinstance(block, list):
            mask_block = block[0].mask
        else:
            mask_block = block.mask

        # if everything is masked
        if np.all(mask_block):
            size = 0
        # if everything is not masked
        elif np.all(mask_block == False):
            size = 1

        # if part masked, part unmasked
        elif np.any(mask_block == False):
            size = 1

            if self.return_3d:
                if mask_block.ndim > 2:
                    mask = mask_block[..., 0]

            elif mask_block.ndim == 1:
                mask = mask_block
            else:
                mask = mask_block[..., 0]

            if isinstance(block, list) and self.return_3d is False:
                block = [b[~mask, ...].data for b in block]
        return block

    def get_random_block(self, random_state=None):
        """
        Get a random block from the raster.

        Parameters
        ------------
        random_state : int, optional (default=None)
            If int, random_state is the seed used by the random number generator.
            If None, the random number generator is the RandomState instance used by numpy np.random.
        """
#        mask = np.array([True])

        np.random.seed(random_state)
        rdm = np.random.permutation(np.arange(self.n_blocks))
        idx = 0
        size = 0

        while size == 0:
            tmp = self.get_block(block_number=rdm[idx], return_with_mask=True)

            if isinstance(tmp, list):
                mask_block = tmp[0].mask
            else:
                mask_block = tmp.mask

            # if everything is masked
            if np.all(mask_block):
                size = 0
            # if everything is not masked
            elif np.all(mask_block == False):
                size = 1
                tmp = tmp

            # if part masked, part unmasked
            elif np.any(mask_block == False):
                size = 1

                if self.return_3d:
                    if mask_block.ndim > 2:
                        mask = mask_block[..., 0]

                elif mask_block.ndim == 1:
                    mask = mask_block
                else:
                    mask = mask_block[..., 0]

                if isinstance(tmp, list) and self.return_3d is False:
                    tmp = [b[~mask, ...].data for b in tmp]

            idx += 1

        return tmp

    def reshape_ndim(self, x):
        """
        Reshape array with at least one band.

        Parameters
        ----------
        x : numpy.ndarray, shape [n_pixels, n_features] or shape [n_pixels].

        Returns
        -------
        x : numpy.ndarray, shape [n_pixels, n_features].

        """
        if x.ndim == 0:
            x = x.reshape(-1, 1)

        if x.ndim == self.return_3d + 1:
            x = x.reshape(*x.shape, 1)

        return x

    def read_band_per_band(self):
        """
        Yields each whole band as np masked array (so with masked data)
        """
        for nRaster in range(len(self.opened_images)):
            nb = self.opened_images[nRaster].RasterCount
            for n in range(1, nb + 1):
                band = self.opened_images[nRaster].GetRasterBand(n)
                band = band.ReadAsArray()
                if self.mask:
                    mask = np.asarray(
                        self.opened_mask.GetRasterBand(1).ReadAsArray(), dtype=bool)
                    band = np.ma.MaskedArray(band, mask=~mask)
                else:
                    band = np.ma.MaskedArray(
                        band, mask=np.where(
                            band == self.nodata, True, False))

                yield band

    def read_block_per_block(self):
        """
        Yield each block.
        """
        for block in range(self.n_blocks):
            yield self.get_block(block)

    def custom_block_size(self, x_block_size=False, y_block_size=False):
        """
        Define custom block size for reading and writing the raster.

        Parameters
        ----------
        y_block_size : float or int, default False.
            IF int, number of rows per block.
            If -1, means all the rows.
            If float, value must be between 0 and 1, such as 1/3.
        x_block_size : float or int, default False.
            If int, number of columns per block.
            If -1, means all the columns.
            If float, value must be between 0 and 1, such as 1/3.
        """

        if y_block_size:
            if y_block_size == -1:
                self.y_block_size = self.n_lines
            elif isinstance(y_block_size, float):
                self.y_block_size = int(np.ceil(self.n_lines * y_block_size))
            else:
                self.y_block_size = y_block_size
        else:
            self.y_block_size = self.block_sizes[1]
        if x_block_size:
            if x_block_size == -1:
                self.x_block_size = self.n_columns
            elif isinstance(x_block_size, float):
                self.x_block_size = int(np.ceil(self.n_columns * x_block_size))
            else:
                self.x_block_size = x_block_size
        else:
            self.x_block_size = self.block_sizes[0]

        self.n_blocks = np.ceil(self.n_lines / self.y_block_size).astype(
            int) * np.ceil(self.n_columns / self.x_block_size).astype(int)
        self.block_sizes = [self.x_block_size, self.y_block_size]

        self.n_y_blocks = len(
            [i for i in range(0, self.n_lines, self.y_block_size)])
        self.n_x_blocks = len(
            [i for i in range(0, self.n_columns, self.x_block_size)])

        # to compute memory size needed in run
        self.size = self.y_block_size * self.x_block_size
        if self.verbose:
            push_feedback('Total number of blocks : %s' % self.n_blocks)

    # =============================================================================
    #           Begin_process_block function for RasterMath
    #               Function is outside of class in order to be used with Parallel
    # =============================================================================
    @staticmethod
    def _process_block(fun, kwargs, block, n_bands, nodata,
                       np_dtype, return_3d, idx_block):
        """
        Private function to compute external function per block.
        In order to save with the size as the input block, this function need the input mask.

        Parameters
        ----------
        fun : function

        kwargs : kwargs
            False, or dict of kwargs.
        block : np.ndarray or np.ma.MaskedArray
            The function will compute using the given block
        nodata : nodata, int or float
            No Data value is some pixels are masked
        np_dtype : numpy datatype
            Numpy datatype for the output array
        return_3d : boolean
            2d-array or 3d-array.

        Returns
        -------
        res : np.ndarray
            The block to write

        """
        tmp_arr = True

        if isinstance(block, list):
            mask_block = block[0].mask
        else:
            mask_block = block.mask

        # if everything is masked
        if np.all(mask_block):
            mask = True
            out_block = nodata
        # if everything is not masked
        elif np.all(mask_block == False):
            mask = False

        # if part masked, part unmasked
        elif np.any(mask_block == False):
            tmp_arr = mask_block.size
            if return_3d:
                if mask_block.ndim > 2:
                    mask = mask_block[..., 0]
            elif mask_block.ndim == 1:
                mask = mask_block
            else:
                mask = mask_block[..., 0]

        # if block is not fully masked
        if mask is not True:

            # if no mask, we send the np.ndarray only
            if mask is False:
                if isinstance(block, list):
                    block = [b.data for b in block]
                else:
                    block = block.data
                if kwargs:
                    out_block = fun(block, **kwargs)
                else:
                    out_block = fun(block)

            # if part is masked
            else:
                # if 3d we send the np.ma.MaskedArray
                if return_3d:
                    if kwargs:
                        out_block = fun(block, **kwargs)
                    else:
                        out_block = fun(block)

                    if out_block.ndim == 1:
                        out_block = np.expand_dims(out_block, 2)
                    out_block[mask, ...] = nodata

                # if 2d, we send only the np.ndarray without masked data
                else:
                    # create empty array with nodata value
                    if mask_block.ndim > 1:
                        mbs = mask_block.shape[:-1]
                    else:
                        mbs = mask_block.shape

                    out_block_shape = list(mbs)
                    out_block_shape.append(n_bands)
                    out_block = np.full(out_block_shape, nodata, np_dtype)

                    if isinstance(block, list):
                        block = [b[~mask, ...].data for b in block]
                        if kwargs:
                            tmp_arr = fun(block, **kwargs)
                        else:
                            tmp_arr = fun(block)
                    else:
                        if kwargs:
                            tmp_arr = fun(block[~mask].data, **kwargs)
                        else:
                            tmp_arr = fun(block[~mask, ...].data)

                    if tmp_arr.ndim == 1:
                        tmp_arr = tmp_arr.reshape(-1, 1)

                    out_block[~mask, ...] = tmp_arr

        return out_block

    # =============================================================================
    #           End of _process_block function
    # =============================================================================

    def run(self, memory_size='1G'):
        """
        Execute and Write output according to given functions.

        Parameters
        ----------
        memory_size : str, optional (default='1G')
            Maximun size of ram the program can use to store blocks and results in memory.
            Support : M,G,T for Mo,Go, or To
            Example : '256M', '1G', '8G' or '1T'.
            Put -1 or False to use all the free memory.

        Returns
        -------
        None
        """
        # Initalize the run
        self._position = 0

        if memory_size == -1 or memory_size is False:
            memory_to_use = virtual_memory().available
        else:
            try:
                size_value = memory_size[:-1]
                if memory_size[-1].capitalize() == 'K':
                    memory_to_use = 1049 * int(size_value)
                if memory_size[-1].capitalize() == 'M':
                    memory_to_use = 1048576 * int(size_value)
                elif memory_size[-1].capitalize() == 'G':
                    memory_to_use = 1073741824 * int(size_value)
                elif memory_size[-1].capitalize() == 'T':
                    memory_to_use = 1099511627776 * int(size_value)
            except BaseException:
                raise ValueError(
                    ' {} is not a valid value. Use for example 100M, 10G or 1T.'.format(memory_size))

        # Compute the size needed in memory for each output function and input
        # block
        items_size = self.itemsize
        for i in self._outputs:
            items_size += i['itemsize']
        minimun_memory = items_size * self.x_block_size * self.y_block_size
        length = min(int(memory_to_use / minimun_memory), self.n_blocks)

        if length == 0:
            minimum_memory_kb = np.ceil(
                np.divide(minimun_memory, 1049)).astype(int)
            raise MemoryError(
                'Not enought memory. For one block, you need at least {}{}'.format(
                    minimum_memory_kb, 'K'))
        if self.verbose:

            if length > self.n_blocks:
                total = self.n_blocks
            else:
                total = length
            print('Batch processing ({} blocks using {}Mo of ram)'.format(
                int(total), np.ceil(minimun_memory * total / 1048576).astype(int)))

            self.pb = ProgressBar(self.n_blocks, message=self.message)
            self._position = 0

        for i in range(0, self.n_blocks, length):

            if i <= self.n_blocks - length:
                idx_blocks = np.arange(i, i + length)
            else:
                idx_blocks = np.arange(i, self.n_blocks)

            for idx_output, output in enumerate(self._outputs):

                function = output['function']
                kwargs = output['kwargs']

                if self.n_jobs > 1:
                    res = Parallel(
                        self.n_jobs)(
                        delayed(
                            self._process_block)(
                            function,
                            kwargs,
                            self.get_block(
                                idx_block,
                                return_with_mask=True),
                            output['n_bands'],
                            output['nodata'],
                            output['np_type'],
                            self.return_3d,
                            idx_block) for idx,
                        idx_block in enumerate(idx_blocks))
                    if self.verbose:
                        self._position += total / len(self._outputs)
                        self.pb.add_position(self._position)

                else:
                    res = []
                    for idx, idx_block in enumerate(idx_blocks):

                        res.append(
                            self._process_block(
                                function,
                                kwargs,
                                self.get_block(
                                    idx_block,
                                    return_with_mask=True),
                                output['n_bands'],
                                output['nodata'],
                                output['np_type'],
                                self.return_3d,
                                idx_block))
                        if self.verbose:
                            self._position += 1 / len(self._outputs)
                            self.pb.add_position(self._position)

                for idx_block, block in enumerate(res):
                    self.write_block(block, idx_blocks[idx_block], idx_output)

                self._outputs[idx_output]['gdal_object'].FlushCache()

        # delete output gdal object
        for fun in self._outputs:
            if fun['nodata'] is not False:
                band = fun['gdal_object'].GetRasterBand(1)
                band.SetNoDataValue(fun['nodata'])
                band.FlushCache()
            fun['gdal_object'] = None

        # no more thing to do
        if self.verbose:
            self.pb.add_position(self.n_blocks)

    def write_block(self, block, idx_block, idx_func):
        """
        Write a block at a position on a output image

        Parameters
        ----------
        idx_block : int.
            List of indexes of all blocks
        tab_blocks : numpy tab.
            List of values or tab that will be written in the output image
        idx_func : int
            function's index

        Returns
        -------
        None.
        """

        coords = self.get_block_coords(idx_block)
        for ind in range(self._outputs[idx_func]['n_bands']):
            # write result band per band
            indGdal = ind + 1

            curBand = self._outputs[idx_func]['gdal_object'].GetRasterBand(
                indGdal)

            # if int or float value, write it in each pixel of the block
            if isinstance(block, (float, int)):
                resToWrite = np.full((coords[3], coords[2]), block)

            else:
                # to be sure to have 2 or 3 dim
                resToWrite = self.reshape_ndim(block)[..., ind]
                if resToWrite.ndim <= 1:
                    resToWrite = self.reshape_ndim(
                        resToWrite).reshape(coords[3], coords[2])

            curBand.WriteArray(resToWrite, coords[0], coords[1])

            # rm FlushCache to speed process
            # curBand.FlushCache()


def sample_extraction(
        in_image,
        in_vector,
        out_vector,
        unique_fid=None,
        band_prefix=None,
        verbose=1):
    """
    Extract centroid from shapefile according to the raster, and extract band value if band_prefix is given.

    This script is available via terminal by entering : `mtb_SampleExtraction`.

    Parameters
    ----------
    in_image : str.
        A filename or path of a raster file.
        It could be any file that GDAL can open.
    in_vector : str.
        A filename or path corresponding to a vector file.
        It could be any file that GDAL/OGR can open.
    out_vector : str.
        Extension will be used to select driver. Please use one of them : ['gpkg','sqlite','shp','netcdf','gpx'].
    unique_fid : str, optional (default=None).
        If None, will add a field called 'uniquefid' in the output vector.
    band_prefix : str, optional (default=None).
        If band_prefix (e.g. 'band'), will extract values from raster.
    """

    def _pixel_location_from_centroid(coords, geo_transform):
        """
        Convert XY coords into the centroid of a pixel

        Parameters
        --------
        coords : arr or list.
            X is coords[0], Y is coords[1].
        geo_transform : list.
            List got from gdal.Open(inRaster).GetGeoTransform() .
        """
        newX = geo_transform[1] * (coords[0] + 0.5) + \
            geo_transform[0]

        newY = geo_transform[5] * (coords[1] + 0.5) + geo_transform[3]
        return [newX, newY]

    if unique_fid is None:
        unique_fid = 'uniquefid'
        if verbose:
            push_feedback("Adding 'uniquefid' field to the original vector.")
        _add_vector_unique_fid(
            in_vector, unique_fid, verbose=verbose)

    if verbose:
        push_feedback("Extract values from raster...")
    X, Y, coords = extract_ROI(
        in_image, in_vector, unique_fid, get_pixel_position=True, verbose=verbose)

    geo_transform = gdal.Open(in_image).GetGeoTransform()

    centroid = [_pixel_location_from_centroid(
        coord, geo_transform) for coord in coords]
    # init outLayer
    if np.issubdtype(X.dtype, np.integer):
        try:
            dtype = ogr.OFTInteger64
        except BaseException:
            dtype = ogr.OFTInteger
    else:
        dtype = ogr.OFTReal
    outLayer = _create_point_layer(
        in_vector, out_vector, unique_fid, dtype=dtype, verbose=verbose)
    if verbose:
        outLayer._add_total_points(len(centroid))

    if verbose:
        push_feedback("Adding each centroid to {}...".format(out_vector))
    for idx, xy in enumerate(centroid):
        try:
            curY = Y[idx][0]
        except BaseException:
            curY = Y[idx]
        if curY != 0:
            if band_prefix is None:
                outLayer._add_point_to_layer(xy, curY)
            else:
                outLayer._add_point_to_layer(xy, curY, X[idx], band_prefix)

    outLayer.close_layer()


class _create_point_layer:
    def __init__(self, in_vector, out_vector, unique_id_field,
                 dtype=ogr.OFTInteger, verbose=1):
        """
        Create a vector layer as point type.

        Parameters
        ------------
        in_vector : str.
            A filename or path corresponding to a vector file.
            It could be any file that GDAL/OGR can open.
        out_vector : str.
            Outvector. Extension will be used to select driver. Please use one of them : ['gpkg','sqlite','shp','netcdf','gpx'].
        unique_fid : str, optional (default=None).
            If None, will add a field called 'uniquefid' in the output vector.
        dtype : int, optional (default=ogr.OFTInteger)
            the ogr datatype.
        verbose : bool or int, optional (default=True).
            The higher is the int verbose, the more it will returns informations.

        Methods
        ----------
        _add_total_points(nSamples): int.
            Will generate progress bar.
        _add_point_to_layer(coords): list,arr.
            coords[0] is X, coords[1] is Y.
        closeLayer():
            Close the layer.
        """
        self.verbose = verbose
        self._dtype = dtype
        # load inVector
        self.inData = ogr.Open(in_vector, 0)
        self.inLyr = self.inData.GetLayerByIndex(0)
        srs = self.inLyr.GetSpatialRef()

        # create outVector
        self.driver_name = get_ogr_driver_from_filename(out_vector)
        driver = ogr.GetDriverByName(self.driver_name)
        self.outData = driver.CreateDataSource(out_vector)

        # finish  outVector creation
        self.outLyr = self.outData.CreateLayer('centroid', srs, ogr.wkbPoint)
        self.outLyrDefinition = self.outLyr.GetLayerDefn()

        # initialize variables
        self.idx = 0
        self.lastPosition = 0
        self.nSamples = None
        if self.driver_name == 'SQLITE' or self.driver_name == 'GPKG':
            self.outLyr.StartTransaction()

        self.unique_id_field = unique_id_field

        # Will generate unique_ID_and_FID when copying vector
        self.unique_ID_and_FID = False
        self.addBand = False

    def _add_band_value(self, bandPrefix, nBands):
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
            self.outLyr.CreateField(ogr.FieldDefn(field, self._dtype))
        self.addBand = True

    def _add_total_points(self, nSamples):
        """
        Adding the total number of points will show a progress bar.

        Parameters
        --------
        nSamples : int.
            The number of points to be added (in order to have a progress bar. Will not affect the processing if bad value is put here.)
        """
        self.nSamples = nSamples
        self.pb = ProgressBar(nSamples, 'Adding points... ')

    def _add_point_to_layer(
            self,
            coords,
            uniqueIDValue,
            band_value=None,
            band_prefix=None):
        """
        Parameters
        -------
        coords : list, or arr.
            X is coords[0], Y is coords[1]
        uniqueIDValue : int.
            Unique ID Value to retrieve the value from fields
        band_value : None, or arr.
            If array, should have the same size as the number of bands defined in addBandsValue function.
        """
        if self.verbose:
            if self.nSamples:
                currentPosition = int(self.idx + 1)
                if currentPosition != self.lastPosition:
                    self.pb.add_position(self.idx + 1)
                    self.lastPosition = currentPosition

        if self.unique_ID_and_FID is False:
            self._update_arr_according_to_vector()

        # add Band to list of fields if needed
        if band_value is not None and self.addBand is False:
            self._add_band_value(band_prefix, band_value.shape[0])

        point = ogr.Geometry(ogr.wkbPoint)
        point.SetPoint(0, coords[0], coords[1])
        featureIndex = self.idx
        feature = ogr.Feature(self.outLyrDefinition)
        feature.SetGeometry(point)
        feature.SetFID(featureIndex)

        # Retrieve inVector FID
        FID = self.uniqueFIDs[np.where(np.asarray(
            self.uniqueIDs, dtype=np.int) == int(uniqueIDValue))[0][0]]

        featUpdates = self.inLyr.GetFeature(int(FID))
        for f in self.fields:
            if f != 'ogc_fid':
                feature.SetField(f, featUpdates.GetField(f))
                if self.addBand is True:
                    for idx, f in enumerate(self.nBandsFields):
                        feature.SetField(f, int(band_value[idx]))

        self.outLyr.CreateFeature(feature)
        self.idx += 1

    def _update_arr_according_to_vector(self):
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
            uID = feat.GetField(self.unique_id_field)
            uFID = feat.GetFID()
            self.uniqueIDs.append(uID)
            self.uniqueFIDs.append(uFID)
        self.unique_ID_and_FID = True

    def close_layer(self):
        """
        Once work is done, close all layers.
        """
        if self.driver_name == 'SQLITE' or self.driver_name == 'GPKG':
            self.outLyr.CommitTransaction()
        self.inData.Destroy()
        self.outData.Destroy()


def get_distance_matrix(in_image, in_vector, field=False, verbose=False):
    """
    Return for each pixel, the distance one-to-one to the other pixels listed in the vector.

    Parameters
    ----------
    in_image : str.
        Path of the raster file where the vector file will be rasterize.
    in_vector : str.
        Path of the vector file to rasterize.
    field : str or False, optional (default=False).
        Name of the vector field to extract the value (must be float or integer).

    Returns
    --------
    distance_matrix : array of shape (nSamples,nSamples)
    label : array of shape (nSamples)
    """
    if field is not False:
        only_pixel_position = False
    else:
        only_pixel_position = True

    coords = extract_ROI(
        in_image,
        in_vector,
        field,
        get_pixel_position=True,
        only_pixel_position=only_pixel_position,
        verbose=verbose)
    from scipy.spatial import distance
    if field:
        label = coords[1]
        coords = coords[2]

    distance_matrix = np.asarray(distance.cdist(
        coords, coords, 'euclidean'), dtype=np.uint64)

    if field:
        return distance_matrix, label
    else:
        return distance_matrix


def get_ogr_driver_from_filename(fileName):
    """
    Return driver name used in OGR accoriding to the extension of the vector.

    Parameters
    ----------
    fileName : str.
        Path of the vector with extension.

    Returns
    -------
    driverName : str
        'SQLITE', 'GPKG', 'ESRI Shapefile'...

    Examples
    --------
    >>> mtb.processing.get_ogr_driver_from_filename('goVegan.gpkg')
    'GPKG'
    >>> mtb.processing.get_ogr_driver_from_filename('stopEatingAnimals.shp')
    'ESRI Shapefile'
    """
    extensions = ['sqlite', 'shp', 'netcdf', 'gpx', 'gpkg']
    driversName = ['SQLITE', 'ESRI Shapefile', 'netCDF', 'GPX', 'GPKG']

    fileName, ext = os.path.splitext(fileName)

    if ext[1:] not in extensions:
        msg = 'Your extension {} is not recognized as a valid extension for saving shape.\n'.format(
            ext)
        msg = msg + 'Supported extensions are ' + str(driversName) + '\n'
        msg = msg + 'We recommend you to use \'sqlite\' extension.'
        raise Warning(msg)
    else:
        driverIdx = [x for x, i in enumerate(extensions) if i == ext[1:]][0]
        driverName = driversName[driverIdx]
        return driverName


def read_vector_values(vector, *args, **kwargs):
    """
    Read values from vector. Will list all fields beginning with the roiprefix 'band-' for example.

    Parameters
    ----------
    vector : str
        Vector path ('myFolder/class.shp',str).
    *args : str
        Field name containing the field to extract values from (i.e. 'class', str).
    **kwargs : arg
        - band_prefix = 'band-' which is the common suffix listing the spectral values (i.e. band_prefix = 'band-').
        - get_features = True, will return features in one list AND spatial Reference.

    Returns
    -------
    List values, same length as number of parameters.
    If band_prefix as parameters, will return one array with n dimension.

    See also
    ---------
    museotoolbox.processing.extract_ROI: extract raster values from vector file.

    Examples
    ---------
    >>> from museotoolbox.datasets import load_historical_data
    >>> _,vector=load_historical_data()
    >>> Y = read_vector_values(vector,'Class')
    array([1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 5, 4, 5, 3, 3, 3], dtype=int32)
    >>> Y,fid = read_vector_values(vector,'Class','uniquefid')
    (array([1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 4, 5, 4, 5, 3, 3, 3], dtype=int32),
     array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17], dtype=int32))
    """

    try:
        file = ogr.Open(vector)
        lyr = file.GetLayer()
    except BaseException:
        raise Exception("Can't open {} file".format(vector))

    # get all fields and save only roiFields
    ldefn = lyr.GetLayerDefn()
    listFields = []

    # add kwargs
    extractBands = False
    get_features = False
    if kwargs:
        # check if need to extract bands from vector
        if 'band_prefix' in kwargs.keys():
            extractBands = True
            band_prefix = kwargs['band_prefix']
        # check if need to extract features from vector
        if 'get_features' in kwargs.keys():
            get_features = kwargs['get_features']

    if extractBands:
        bandsFields = []

    # if get_features, save Spatial Reference and Features
    if get_features:
        srs = lyr.GetSpatialRef()
        features = []

    # List available fields
    for n in range(ldefn.GetFieldCount()):
        fdefn = ldefn.GetFieldDefn(n)
        if fdefn.name is not listFields:
            listFields.append(fdefn.name)
        if extractBands:
            if fdefn.name.startswith(band_prefix):
                bandsFields.append(fdefn.name)

    if len(kwargs) == 0 and len(args) == 0:
        raise ValueError('These fields are available : {}'.format(listFields))
    else:

        if extractBands and len(bandsFields) == 0:
            raise ValueError(
                'Band prefix field "{}" do not exists. These fields are available : {}'.format(
                    band_prefix, listFields))

        # Initialize empty arrays
        if len(args) > 0:  # for single fields
            ROIlevels = [np.zeros(lyr.GetFeatureCount()) for i in args]

        if extractBands:  # for band_prefix
            ROIvalues = np.zeros(
                [lyr.GetFeatureCount(), len(bandsFields)], dtype=np.int32)

        # Listing each feature and store to array
        for i, feature in enumerate(lyr):
            if extractBands:
                for j, band in enumerate(bandsFields):
                    feat = feature.GetField(band)
                    if i == 0:
                        ROIvalues.astype(type(feat))

                    ROIvalues[i, j] = feat
            if len(args) > 0:
                try:
                    for a in range(len(args)):
                        feat = feature.GetField(args[a])
                        if i == 0:
                            ROIlevels[a] = ROIlevels[a].astype(type(feat))
                        ROIlevels[a][i] = feature.GetField(args[a])
                except BaseException:
                    raise ValueError(
                        "Field \"{}\" do not exists. These fields are available : {}".format(
                            args[a], listFields))
            if get_features:
                features.append(feature)

        # Initialize return
        fieldsToReturn = []

        # if bandPrefix
        if extractBands:
            fieldsToReturn.append(ROIvalues)

        # if single fields
        if len(args) > 0:
            for i in range(len(args)):
                fieldsToReturn.append(ROIlevels[i])

        # if features
        if get_features:
            fieldsToReturn.append(features)
            fieldsToReturn.append(srs)
        # if 1d, to turn single array
        if len(fieldsToReturn) == 1:
            fieldsToReturn = fieldsToReturn[0]

        return fieldsToReturn


def _add_vector_unique_fid(in_vector, unique_field='uniquefid', verbose=True):
    """
    Add a field in the vector with an unique value
    for each of the feature.

    Parameters
    -----------
    inVector : str
        Path of the vector file.
    uniqueField : str
        Name of the field to create
    verbose : bool or int, default True.
    Returns
    --------
    None

    Examples
    ---------
    >>> _add_vector_unique_fid('myDB.gpkg',uniqueField='polygonid')
    Adding polygonid [########################################]100%
    """
    if verbose:
        pB = ProgressBar(100, message='Adding ' + unique_field)

    driver_name = get_ogr_driver_from_filename(in_vector)
    inDriver = ogr.GetDriverByName(driver_name)
    inSrc = inDriver.Open(in_vector, 1)  # 1 for writable
    inLyr = inSrc.GetLayer()       # get the layer for this datasource
    inLyrDefn = inLyr.GetLayerDefn()

    if driver_name == 'SQLITE' or driver_name == 'GPKG':
        inLyr.StartTransaction()

    listFields = []
    for n in range(inLyrDefn.GetFieldCount()):
        fdefn = inLyrDefn.GetFieldDefn(n)
        if fdefn.name is not listFields:
            listFields.append(fdefn.name)
    if unique_field in listFields:
        if verbose > 0:
            print(
                'Field \'{}\' is already in {}'.format(
                    unique_field, in_vector))
        inSrc.Destroy()
    else:
        newField = ogr.FieldDefn(unique_field, ogr.OFTInteger)
        newField.SetWidth(20)
        inLyr.CreateField(newField)

        FIDs = [feat.GetFID() for feat in inLyr]

        ThisID = 1

        for idx, FID in enumerate(FIDs):
            if verbose:
                pB.add_position(idx / len(FIDs) + 1 * 100)
            feat = inLyr.GetFeature(FID)
            #ThisID = int(feat.GetFGetFeature(feat))
            # Write the FID to the ID field
            feat.SetField(unique_field, int(ThisID))
            inLyr.SetFeature(feat)              # update the feature
            # inLyr.CreateFeature(feat)
            ThisID += 1

        if driver_name == 'SQLITE' or driver_name == 'GPKG':
            inLyr.CommitTransaction()
        inSrc.Destroy()


def _reshape_ndim(X):
    """
    Reshape ndim of X to have at least 2 dimensions

    Parameters
    -----------
    X : np.ndarray
        array.

    Returns
    --------
    X : np.ndarray
        Returns array with a least 2 dimensions.

    Examples
    ---------
    >>> X = np.arange(5,50)
    >>> X.shape
    (45,)
    >>> _reshape_ndim(X).shape
    (45, 1)
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X
