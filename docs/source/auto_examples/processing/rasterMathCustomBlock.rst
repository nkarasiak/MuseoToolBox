.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_processing_rasterMathCustomBlock.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_processing_rasterMathCustomBlock.py:


rasterMath with custom window/block size (and with 3 dimensions)
=================================================================

Tips to use rasterMath by defining its block size and to receive
a full block (not a array with one pixel per row.)


Import librairies
-------------------------------------------


.. code-block:: default


    from museotoolbox.processing import RasterMath
    from museotoolbox import datasets
    from matplotlib import pyplot as plt







Load HistoricalMap dataset
-------------------------------------------


.. code-block:: default


    raster,vector = datasets.load_historical_data()







Initialize rasterMath with raster
------------------------------------


.. code-block:: default


    # Set return3d to True to have full block size (not one pixel per row)

    rM = RasterMath(raster,return_3d=True)

    print(rM.get_random_block().shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 15
    (256, 256, 3)


Comparing different block size (%, fixed, full block)
-------------------------------------------------------

You can define block by percentage of the whole width/height


.. code-block:: default


    rM.custom_block_size(1/2,1/2) 
    print(rM.get_random_block().shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 4
    (283, 527, 3)


Or by fixed window 


.. code-block:: default


    rM.custom_block_size(50,100) # width divided every 50 pixel and height every 100
    print(rM.get_random_block().shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 132
    (100, 50, 3)


To have the full image (one block)


.. code-block:: default


    rM.custom_block_size(-1,-1) # to have the full image





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 1


To have block width divided by 4 and height by 2


.. code-block:: default


    rM.custom_block_size(1/4,1/2)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 8


Define block size for output raster
-------------------------------------


.. code-block:: default


    raster_parameters = rM.get_raster_parameters()

    print('Default parameters are '+str(raster_parameters))


    # to do before adding the function

    rM.custom_block_size(256,256) # custom for reading AND writing the output
    #raster_parameters = ['COMPRESS=DEFLATE']
    #rM.customRasterParameters(raster_parameters)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Default parameters are ['BIGTIFF=IF_SAFER', 'NUM_THREADS=3', 'COMPRESS=PACKBITS']
    Total number of blocks : 15


now add a function to just return the same raster


.. code-block:: default


    returnSameImage  = lambda x : x
    rM.add_function(returnSameImage,'/tmp/testcustomblock.tif')
    rM.run()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Using datatype from numpy table : uint8.
    Detected 3 bands for function <lambda>.
    rasterMath... [........................................]0%    rasterMath... [##......................................]6%    rasterMath... [#####...................................]13%    rasterMath... [########................................]20%    rasterMath... [##########..............................]26%    rasterMath... [#############...........................]33%    rasterMath... [################........................]40%    rasterMath... [##################......................]46%    rasterMath... [#####################...................]53%    rasterMath... [########################................]60%    rasterMath... [##########################..............]66%    rasterMath... [#############################...........]73%    rasterMath... [################################........]80%    rasterMath... [##################################......]86%    rasterMath... [#####################################...]93%    rasterMath... [########################################]100%
    Saved /tmp/testcustomblock.tif using function <lambda>


check block size of new raster


.. code-block:: default


    rMblock = RasterMath('/tmp/testcustomblock.tif')
    print(rMblock.block_sizes)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 15
    [256, 256]


Plot blocks


.. code-block:: default


    n_row,n_col = 2,4
    rM.custom_block_size(1/n_col,1/n_row)

    fig=plt.figure(figsize=(12,6),dpi=150)

    for idx,tile in enumerate(rM.read_block_per_block()):
        fig.add_subplot(n_row,n_col,idx+1)
        plt.title('block %s' %(idx+1))
        plt.imshow(tile)
    plt.show()




.. image:: /auto_examples/processing/images/sphx_glr_rasterMathCustomBlock_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 8



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.886 seconds)


.. _sphx_glr_download_auto_examples_processing_rasterMathCustomBlock.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: rasterMathCustomBlock.py <rasterMathCustomBlock.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: rasterMathCustomBlock.ipynb <rasterMathCustomBlock.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
