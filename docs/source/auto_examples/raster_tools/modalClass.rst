.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_raster_tools_modalClass.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_raster_tools_modalClass.py:


Modal class and number of agreements
===============================================================

Create a raster with the modal class and the number of agreements.



Import librairies
-------------------------------------------



.. code-block:: python


    import museotoolbox as mtb
    from scipy import stats
    import numpy as np






Load HistoricalMap dataset
-------------------------------------------



.. code-block:: python


    raster,vector = mtb.datasets.historicalMap()







Initialize rasterMath with raster
-----------------------------------------



.. code-block:: python


    ########
    # In case you want to add a mask
    mask = '/tmp/maskFromPolygons.tif'
    mtb.raster_tools.rasterMaskFromVector(vector,raster,mask,invert=False)

    rM = mtb.raster_tools.rasterMath(raster,inMaskRaster=mask)

    print(rM.getRandomBlock())




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 15
    [[-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]
     [-- -- --]]


Let's suppose you want compute the modal classification between several predictions
The first band will be the most predicted class, and the second the number of times it has been predicted.



.. code-block:: python



    x = rM.getRandomBlock()

    def modalClass(x):
        tmp = stats.mode(x, axis=-1)
        tmpStack = np.column_stack((tmp[0], tmp[1])).astype(np.int16)
        return tmpStack


    rM.addFunction(modalClass,outRaster='/tmp/modal.tif',outNoData=0)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Using datatype from numpy table : int16
    Detected 2 band(s) for function modalClass.


Run the script



.. code-block:: python


    rM.run()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    rasterMath...  [........................................]0%    rasterMath...  [##......................................]6%    rasterMath...  [#####...................................]13%    rasterMath...  [########................................]20%    rasterMath...  [##########..............................]26%    rasterMath...  [#############...........................]33%    rasterMath...  [################........................]40%    rasterMath...  [##################......................]46%    rasterMath...  [#####################...................]53%    rasterMath...  [########################................]60%    rasterMath...  [##########################..............]66%    rasterMath...  [#############################...........]73%    rasterMath...  [################################........]80%    rasterMath...  [##################################......]86%    rasterMath...  [#####################################...]93%    rasterMath...  [########################################]100%
    Saved /tmp/modal.tif using function modalClass


Plot result



.. code-block:: python


    import gdal
    from matplotlib import pyplot as plt 

    src = gdal.Open('/tmp/modal.tif')
    plt.imshow(src.ReadAsArray()[0,:,:])



.. image:: /auto_examples/raster_tools/images/sphx_glr_modalClass_001.png
    :class: sphx-glr-single-img




**Total running time of the script:** ( 0 minutes  2.762 seconds)


.. _sphx_glr_download_auto_examples_raster_tools_modalClass.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: modalClass.py <modalClass.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: modalClass.ipynb <modalClass.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
