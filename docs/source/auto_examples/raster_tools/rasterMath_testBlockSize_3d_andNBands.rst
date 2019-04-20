.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_raster_tools_rasterMath_testBlockSize_3d_andNBands.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_raster_tools_rasterMath_testBlockSize_3d_andNBands.py:


Tests rasterMath with full block or stacken and custom block size
==================================================================

Test notebook in order to validate code.



Import librairies
-------------------------------------------



.. code-block:: python


    from museotoolbox.raster_tools import rasterMath,rasterMaskFromVector
    from museotoolbox import datasets
    from matplotlib import pyplot as plt
    import numpy as np






Load HistoricalMap dataset
-------------------------------------------



.. code-block:: python


    raster,vector = datasets.historicalMap()







Initialize rasterMath with raster
------------------------------------



.. code-block:: python


    # Set return_3d to True to have full block size (not one pixel per row)
    # Create raster mask to only keep pixel inside polygons.

    rasterMaskFromVector(vector,raster,'/tmp/mask.tif',invert=False)

    for return_3d in [True,False]:
        rM = rasterMath(raster,inMaskRaster='/tmp/mask.tif',return_3d=return_3d)
    
        rM.customBlockSize(200,200) # block of 200x200pixels
    
        print(rM.getRandomBlock().shape)
    
        #######################
        # Plot blocks
    
        x = rM.getRandomBlock()
        def returnFlatten(x):
            try:
                x = x[:,:,0]
            except:
                x = x[:,0].reshape(-1,1)
            return x
        def returnWithOneBandMore(x):
            try:
                x = np.repeat(x,3,axis=2)
            except:
                x= np.repeat(x,3,axis=1)
            return x
    
        rM.addFunction(returnWithOneBandMore,'/tmp/x_repeat_{}.tif'.format(str(return_3d)))
        rM.addFunction(returnFlatten,'/tmp/x_flatten_{}.tif'.format(str(return_3d)))

    
    
        rM.run()
    
    import gdal
    dst = gdal.Open('/tmp/x_flatten_False.tif')
    arr = dst.GetRasterBand(1).ReadAsArray()
    plt.imshow(np.ma.masked_where(arr == np.min(arr), arr))



.. image:: /auto_examples/raster_tools/images/sphx_glr_rasterMath_testBlockSize_3d_andNBands_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 15
    Total number of blocks : 18
    (166, 200, 3)
    Using datatype from numpy table : uint8
    Detected 9 band(s) for function returnWithOneBandMore.
    No data is set to : 0
    Using datatype from numpy table : uint8
    Detected 1 band(s) for function returnFlatten.
    No data is set to : 0
    rasterMath... [........................................]0%    rasterMath... [##......................................]5%    rasterMath... [####....................................]11%    rasterMath... [######..................................]16%    rasterMath... [########................................]22%    rasterMath... [###########.............................]27%    rasterMath... [#############...........................]33%    rasterMath... [###############.........................]38%    rasterMath... [#################.......................]44%    rasterMath... [####################....................]50%    rasterMath... [######################..................]55%    rasterMath... [########################................]61%    rasterMath... [##########################..............]66%    rasterMath... [############################............]72%    rasterMath... [###############################.........]77%    rasterMath... [#################################.......]83%    rasterMath... [###################################.....]88%    rasterMath... [#####################################...]94%    rasterMath... [########################################]100%
    Saved /tmp/x_repeat_True.tif using function returnWithOneBandMore
    Saved /tmp/x_flatten_True.tif using function returnFlatten
    Total number of blocks : 15
    Total number of blocks : 18
    (936, 3)
    Using datatype from numpy table : uint8
    Detected 9 band(s) for function returnWithOneBandMore.
    No data is set to : 0
    Using datatype from numpy table : uint8
    Detected 1 band(s) for function returnFlatten.
    No data is set to : 0
    rasterMath... [........................................]0%    rasterMath... [##......................................]5%    rasterMath... [####....................................]11%    rasterMath... [######..................................]16%    rasterMath... [########................................]22%    rasterMath... [###########.............................]27%    rasterMath... [#############...........................]33%    rasterMath... [###############.........................]38%    rasterMath... [#################.......................]44%    rasterMath... [####################....................]50%    rasterMath... [######################..................]55%    rasterMath... [########################................]61%    rasterMath... [##########################..............]66%    rasterMath... [############################............]72%    rasterMath... [###############################.........]77%    rasterMath... [#################################.......]83%    rasterMath... [###################################.....]88%    rasterMath... [#####################################...]94%    rasterMath... [########################################]100%
    Saved /tmp/x_repeat_False.tif using function returnWithOneBandMore
    Saved /tmp/x_flatten_False.tif using function returnFlatten


**Total running time of the script:** ( 0 minutes  0.864 seconds)


.. _sphx_glr_download_auto_examples_raster_tools_rasterMath_testBlockSize_3d_andNBands.py:


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
