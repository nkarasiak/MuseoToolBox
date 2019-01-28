.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_raster_tools_rasterMath_testBlockSize_3d_andNBands.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_raster_tools_rasterMath_testBlockSize_3d_andNBands.py:


rasterMath tests with full block or stacken and custom block size
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

    for return_3d in [False,True]:
        rM = rasterMath(raster,inMaskRaster='/tmp/mask.tif',return_3d=return_3d)
    
        rM.customBlockSize(100,100) # block of 100x100pixels
    
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
    dst = gdal.Open('/tmp/x_flatten_True.tif')
    arr = dst.GetRasterBand(1).ReadAsArray()
    plt.imshow(np.ma.masked_where(arr == np.min(arr), arr))


.. image:: /auto_examples/raster_tools/images/sphx_glr_rasterMath_testBlockSize_3d_andNBands_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 15
    Total number of blocks : 66
    (10000, 3)
    Using datatype from numpy table : uint8
    Detected 9 band(s) for function returnWithOneBandMore.
    Using datatype from numpy table : uint8
    Detected 1 band(s) for function returnFlatten.
    rasterMath...  [........................................]0%    rasterMath...  [........................................]1%    rasterMath...  [#.......................................]3%    rasterMath...  [#.......................................]4%    rasterMath...  [##......................................]6%    rasterMath...  [###.....................................]7%    rasterMath...  [###.....................................]9%    rasterMath...  [####....................................]10%    rasterMath...  [####....................................]12%    rasterMath...  [#####...................................]13%    rasterMath...  [######..................................]15%    rasterMath...  [######..................................]16%    rasterMath...  [#######.................................]18%    rasterMath...  [#######.................................]19%    rasterMath...  [########................................]21%    rasterMath...  [#########...............................]22%    rasterMath...  [#########...............................]24%    rasterMath...  [##########..............................]25%    rasterMath...  [##########..............................]27%    rasterMath...  [###########.............................]28%    rasterMath...  [############............................]30%    rasterMath...  [############............................]31%    rasterMath...  [#############...........................]33%    rasterMath...  [#############...........................]34%    rasterMath...  [##############..........................]36%    rasterMath...  [###############.........................]37%    rasterMath...  [###############.........................]39%    rasterMath...  [################........................]40%    rasterMath...  [################........................]42%    rasterMath...  [#################.......................]43%    rasterMath...  [##################......................]45%    rasterMath...  [##################......................]46%    rasterMath...  [###################.....................]48%    rasterMath...  [####################....................]50%    rasterMath...  [####################....................]51%    rasterMath...  [#####################...................]53%    rasterMath...  [#####################...................]54%    rasterMath...  [######################..................]56%    rasterMath...  [#######################.................]57%    rasterMath...  [#######################.................]59%    rasterMath...  [########################................]60%    rasterMath...  [########################................]62%    rasterMath...  [#########################...............]63%    rasterMath...  [##########################..............]65%    rasterMath...  [##########################..............]66%    rasterMath...  [###########################.............]68%    rasterMath...  [###########################.............]69%    rasterMath...  [############################............]71%    rasterMath...  [#############################...........]72%    rasterMath...  [#############################...........]74%    rasterMath...  [##############################..........]75%    rasterMath...  [##############################..........]77%    rasterMath...  [###############################.........]78%    rasterMath...  [################################........]80%    rasterMath...  [################################........]81%    rasterMath...  [#################################.......]83%    rasterMath...  [#################################.......]84%    rasterMath...  [##################################......]86%    rasterMath...  [###################################.....]87%    rasterMath...  [###################################.....]89%    rasterMath...  [####################################....]90%    rasterMath...  [####################################....]92%    rasterMath...  [#####################################...]93%    rasterMath...  [######################################..]95%    rasterMath...  [######################################..]96%    rasterMath...  [#######################################.]98%    rasterMath...  [########################################]100%
    Saved /tmp/x_repeat_False.tif using function returnWithOneBandMore
    Saved /tmp/x_flatten_False.tif using function returnFlatten
    Total number of blocks : 15
    Total number of blocks : 66
    (100, 100, 3)
    Using datatype from numpy table : uint8
    Detected 9 band(s) for function returnWithOneBandMore.
    Using datatype from numpy table : uint8
    Detected 1 band(s) for function returnFlatten.
    rasterMath...  [........................................]0%    rasterMath...  [........................................]1%    rasterMath...  [#.......................................]3%    rasterMath...  [#.......................................]4%    rasterMath...  [##......................................]6%    rasterMath...  [###.....................................]7%    rasterMath...  [###.....................................]9%    rasterMath...  [####....................................]10%    rasterMath...  [####....................................]12%    rasterMath...  [#####...................................]13%    rasterMath...  [######..................................]15%    rasterMath...  [######..................................]16%    rasterMath...  [#######.................................]18%    rasterMath...  [#######.................................]19%    rasterMath...  [########................................]21%    rasterMath...  [#########...............................]22%    rasterMath...  [#########...............................]24%    rasterMath...  [##########..............................]25%    rasterMath...  [##########..............................]27%    rasterMath...  [###########.............................]28%    rasterMath...  [############............................]30%    rasterMath...  [############............................]31%    rasterMath...  [#############...........................]33%    rasterMath...  [#############...........................]34%    rasterMath...  [##############..........................]36%    rasterMath...  [###############.........................]37%    rasterMath...  [###############.........................]39%    rasterMath...  [################........................]40%    rasterMath...  [################........................]42%    rasterMath...  [#################.......................]43%    rasterMath...  [##################......................]45%    rasterMath...  [##################......................]46%    rasterMath...  [###################.....................]48%    rasterMath...  [####################....................]50%    rasterMath...  [####################....................]51%    rasterMath...  [#####################...................]53%    rasterMath...  [#####################...................]54%    rasterMath...  [######################..................]56%    rasterMath...  [#######################.................]57%    rasterMath...  [#######################.................]59%    rasterMath...  [########################................]60%    rasterMath...  [########################................]62%    rasterMath...  [#########################...............]63%    rasterMath...  [##########################..............]65%    rasterMath...  [##########################..............]66%    rasterMath...  [###########################.............]68%    rasterMath...  [###########################.............]69%    rasterMath...  [############################............]71%    rasterMath...  [#############################...........]72%    rasterMath...  [#############################...........]74%    rasterMath...  [##############################..........]75%    rasterMath...  [##############################..........]77%    rasterMath...  [###############################.........]78%    rasterMath...  [################################........]80%    rasterMath...  [################################........]81%    rasterMath...  [#################################.......]83%    rasterMath...  [#################################.......]84%    rasterMath...  [##################################......]86%    rasterMath...  [###################################.....]87%    rasterMath...  [###################################.....]89%    rasterMath...  [####################################....]90%    rasterMath...  [####################################....]92%    rasterMath...  [#####################################...]93%    rasterMath...  [######################################..]95%    rasterMath...  [######################################..]96%    rasterMath...  [#######################################.]98%    rasterMath...  [########################################]100%
    Saved /tmp/x_repeat_True.tif using function returnWithOneBandMore
    Saved /tmp/x_flatten_True.tif using function returnFlatten


**Total running time of the script:** ( 0 minutes  17.894 seconds)


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
