.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_raster_tools_rasterMathCustomBlockAndMask.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_raster_tools_rasterMathCustomBlockAndMask.py:


rasterMath with custom block size, mask, and in 3 dimensions
===================================================================

Tips to use rasterMath by defining its block size and to receive
a full block (not a array with one pixel per row.)

Tips : A function readBlockPerBlock() yields each block, without saving results
to a new raster.



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

    rM = rasterMath(raster,inMaskRaster='/tmp/mask.tif',return_3d=True)

    print(rM.getRandomBlock().shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 15
    (256, 256, 3)


Plot blocks



.. code-block:: python

    x = rM.getRandomBlock()

    rM.addFunction(np.mean,'/tmp/mean.tif',axis=2,dtype=np.int16)

    for tile in rM.readBlockPerBlock():
        print(tile)
    #rM.addFunction(returnX,'/tmp/mean.tif')
    rM.run()

    import gdal
    dst = gdal.Open('/tmp/mean.tif')
    arr = dst.GetRasterBand(1).ReadAsArray()
    plt.imshow(np.ma.masked_where(arr == np.min(arr), arr))


.. image:: /auto_examples/raster_tools/images/sphx_glr_rasterMathCustomBlockAndMask_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Using datatype from numpy table : int16
    Detected 1 band(s) for function mean.
    [[[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     ..., 
     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]]
    [[[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     ..., 
     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]]
    [[[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     ..., 
     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]]
    [[[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     ..., 
     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]]
    [[[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     ..., 
     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]]
    [[[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     ..., 
     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]]
    [[[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     ..., 
     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]

     [[-- -- --]
      [-- -- --]
      [-- -- --]
      ..., 
      [-- -- --]
      [-- -- --]
      [-- -- --]]]
    rasterMath...  [........................................]0%    rasterMath...  [##......................................]6%    rasterMath...  [#####...................................]13%    rasterMath...  [########................................]20%    rasterMath...  [##########..............................]26%    rasterMath...  [#############...........................]33%    rasterMath...  [################........................]40%    rasterMath...  [##################......................]46%    rasterMath...  [#####################...................]53%    rasterMath...  [########################................]60%    rasterMath...  [##########################..............]66%    rasterMath...  [#############################...........]73%    rasterMath...  [################################........]80%    rasterMath...  [##################################......]86%    rasterMath...  [#####################################...]93%    rasterMath...  [########################################]100%
    Saved /tmp/mean.tif using function mean


**Total running time of the script:** ( 0 minutes  0.231 seconds)


.. _sphx_glr_download_auto_examples_raster_tools_rasterMathCustomBlockAndMask.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: rasterMathCustomBlockAndMask.py <rasterMathCustomBlockAndMask.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: rasterMathCustomBlockAndMask.ipynb <rasterMathCustomBlockAndMask.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
