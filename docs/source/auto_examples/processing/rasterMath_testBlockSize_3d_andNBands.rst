.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_processing_rasterMath_testBlockSize_3d_andNBands.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_processing_rasterMath_testBlockSize_3d_andNBands.py:


Using rasterMath with 3d block or 2d block
==================================================================

Test notebook to validate code.

Import librairies
-------------------------------------------


.. code-block:: default


    from museotoolbox.processing import RasterMath,image_mask_from_vector
    from museotoolbox import datasets
    from matplotlib import pyplot as plt
    import numpy as np







Load HistoricalMap dataset
-------------------------------------------


.. code-block:: default


    raster,vector = datasets.load_historical_data()







Initialize rasterMath with raster
------------------------------------


.. code-block:: default


    # Set return_3d to True to have full block size (not one pixel per row)
    # Create raster mask to only keep pixel inside polygons.

    image_mask_from_vector(vector,raster,'/tmp/mask.tif',invert=False)

    for return_3d in [True,False]:
        rM = RasterMath(raster,in_image_mask='/tmp/mask.tif',return_3d=return_3d)
    
        rM.custom_block_size(128,128) # block of 200x200pixels
    
        print(rM.get_random_block().shape)
    
        x = rM.get_random_block()
    
        # Returns with only 1 dimension
        returnFlatten = lambda x : x[...,0]
    
        # Returns 3x the original last dimension
        addOneBand = lambda x : np.repeat(x,3,axis=x.ndim-1)
    
        # Add functions to rasterMath
        rM.add_function(addOneBand,'/tmp/x_repeat_{}.tif'.format(str(return_3d)))
        rM.add_function(returnFlatten,'/tmp/x_flatten_{}.tif'.format(str(return_3d)))
    
        rM.run()
    
    from osgeo import gdal
    dst = gdal.Open('/tmp/x_flatten_False.tif')
    arr = dst.GetRasterBand(1).ReadAsArray()
    plt.imshow(np.ma.masked_where(arr == np.min(arr), arr))



.. image:: /auto_examples/processing/images/sphx_glr_rasterMath_testBlockSize_3d_andNBands_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 15
    Total number of blocks : 45
    (128, 128, 3)
    Using datatype from numpy table : uint8.
    Detected 9 bands for function <lambda>.
    No data is set to : 0
    Using datatype from numpy table : uint8.
    Detected 1 band for function <lambda>.
    No data is set to : 0
    rasterMath... [........................................]0%    rasterMath... [........................................]2%    rasterMath... [#.......................................]4%    rasterMath... [##......................................]6%    rasterMath... [###.....................................]8%    rasterMath... [####....................................]11%    rasterMath... [#####...................................]13%    rasterMath... [######..................................]15%    rasterMath... [#######.................................]17%    rasterMath... [########................................]20%    rasterMath... [########................................]22%    rasterMath... [#########...............................]24%    rasterMath... [##########..............................]26%    rasterMath... [###########.............................]28%    rasterMath... [############............................]31%    rasterMath... [#############...........................]33%    rasterMath... [##############..........................]35%    rasterMath... [###############.........................]37%    rasterMath... [################........................]40%    rasterMath... [################........................]42%    rasterMath... [#################.......................]44%    rasterMath... [##################......................]46%    rasterMath... [###################.....................]48%    rasterMath... [####################....................]51%    rasterMath... [#####################...................]53%    rasterMath... [######################..................]55%    rasterMath... [#######################.................]57%    rasterMath... [########################................]60%    rasterMath... [########################................]62%    rasterMath... [#########################...............]64%    rasterMath... [##########################..............]66%    rasterMath... [###########################.............]68%    rasterMath... [############################............]71%    rasterMath... [#############################...........]73%    rasterMath... [##############################..........]75%    rasterMath... [###############################.........]77%    rasterMath... [################################........]80%    rasterMath... [################################........]82%    rasterMath... [#################################.......]84%    rasterMath... [##################################......]86%    rasterMath... [###################################.....]88%    rasterMath... [####################################....]91%    rasterMath... [#####################################...]93%    rasterMath... [######################################..]95%    rasterMath... [#######################################.]97%    rasterMath... [########################################]100%
    Saved /tmp/x_repeat_True.tif using function <lambda>
    Saved /tmp/x_flatten_True.tif using function <lambda>
    Total number of blocks : 15
    Total number of blocks : 45
    (907, 3)
    Using datatype from numpy table : uint8.
    Detected 9 bands for function <lambda>.
    No data is set to : 0
    Using datatype from numpy table : uint8.
    Detected 1 band for function <lambda>.
    No data is set to : 0
    rasterMath... [........................................]0%    rasterMath... [........................................]2%    rasterMath... [#.......................................]4%    rasterMath... [##......................................]6%    rasterMath... [###.....................................]8%    rasterMath... [####....................................]11%    rasterMath... [#####...................................]13%    rasterMath... [######..................................]15%    rasterMath... [#######.................................]17%    rasterMath... [########................................]20%    rasterMath... [########................................]22%    rasterMath... [#########...............................]24%    rasterMath... [##########..............................]26%    rasterMath... [###########.............................]28%    rasterMath... [############............................]31%    rasterMath... [#############...........................]33%    rasterMath... [##############..........................]35%    rasterMath... [###############.........................]37%    rasterMath... [################........................]40%    rasterMath... [################........................]42%    rasterMath... [#################.......................]44%    rasterMath... [##################......................]46%    rasterMath... [###################.....................]48%    rasterMath... [####################....................]51%    rasterMath... [#####################...................]53%    rasterMath... [######################..................]55%    rasterMath... [#######################.................]57%    rasterMath... [########################................]60%    rasterMath... [########################................]62%    rasterMath... [#########################...............]64%    rasterMath... [##########################..............]66%    rasterMath... [###########################.............]68%    rasterMath... [############################............]71%    rasterMath... [#############################...........]73%    rasterMath... [##############################..........]75%    rasterMath... [###############################.........]77%    rasterMath... [################################........]80%    rasterMath... [################################........]82%    rasterMath... [#################################.......]84%    rasterMath... [##################################......]86%    rasterMath... [###################################.....]88%    rasterMath... [####################################....]91%    rasterMath... [#####################################...]93%    rasterMath... [######################################..]95%    rasterMath... [#######################################.]97%    rasterMath... [########################################]100%
    Saved /tmp/x_repeat_False.tif using function <lambda>
    Saved /tmp/x_flatten_False.tif using function <lambda>



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.402 seconds)


.. _sphx_glr_download_auto_examples_processing_rasterMath_testBlockSize_3d_andNBands.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: rasterMath_testBlockSize_3d_andNBands.py <rasterMath_testBlockSize_3d_andNBands.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: rasterMath_testBlockSize_3d_andNBands.ipynb <rasterMath_testBlockSize_3d_andNBands.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
