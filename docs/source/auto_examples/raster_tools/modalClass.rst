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


.. code-block:: default


    import museotoolbox as mtb
    from scipy import stats
    import numpy as np






Load HistoricalMap dataset
-------------------------------------------


.. code-block:: default


    raster,vector = mtb.datasets.historicalMap(low_res=True)







Initialize rasterMath with raster
-----------------------------------------


.. code-block:: default


    ########
    # In case you want to add a mask
    mask = '/tmp/maskFromPolygons.tif'
    mtb.raster_tools.rasterMaskFromVector(vector,raster,mask,invert=False)

    rM = mtb.raster_tools.rasterMath(raster,inMaskRaster=mask)

    print(rM.getRandomBlock())




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 18
    [[148 143  98]
     [144 136  92]
     [145 138  93]
     [154 150 102]
     [158 154 107]
     [145 138  94]
     [127 117  77]
     [136 127  88]
     [132 123  84]
     [126 120  84]
     [120 116  85]
     [113 110  81]
     [105 102  73]
     [100  97  72]
     [ 99  96  77]
     [158 150 107]
     [153 147 102]
     [147 142  95]
     [145 141  95]
     [146 139  96]
     [145 136  96]
     [144 135  97]
     [137 130  91]
     [129 123  87]
     [121 117  86]
     [114 110  81]
     [105 102  72]
     [ 97  94  68]
     [143 137  93]
     [140 134  91]
     [146 139  98]
     [152 144 105]
     [147 140 102]
     [140 133  95]
     [130 124  88]
     [122 118  86]
     [117 113  83]
     [111 107  76]
     [146 139  99]
     [144 138  99]
     [142 135  97]
     [142 136 100]
     [137 131  93]
     [129 124  88]
     [124 119  87]
     [125 120  88]
     [123 118  85]
     [133 128  92]
     [134 129  94]
     [133 128  91]
     [130 125  89]
     [128 123  90]
     [132 126  93]
     [125 122  87]
     [130 127  90]
     [132 128  92]
     [132 126  93]
     [132 128  93]]


Let's suppose you want compute the modal classification between several predictions
The first band will be the most predicted class, and the second the number of times it has been predicted.


.. code-block:: default



    x = rM.getRandomBlock()

    def modalClass(x):
        tmp = stats.mode(x,axis=1)
        tmpStack = np.column_stack((tmp[0], tmp[1])).astype(np.int16)
        return tmpStack

    rM.addFunction(modalClass,outRaster='/tmp/modal.tif',outNoData=0)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Using datatype from numpy table : int16.
    Detected 2 bands for function modalClass.
    No data is set to : 0


Run the script


.. code-block:: default


    rM.run()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    rasterMath... [........................................]0%    rasterMath... [##......................................]5%    rasterMath... [####....................................]11%    rasterMath... [######..................................]16%    rasterMath... [########................................]22%    rasterMath... [###########.............................]27%    rasterMath... [#############...........................]33%    rasterMath... [###############.........................]38%    rasterMath... [#################.......................]44%    rasterMath... [####################....................]50%    rasterMath... [######################..................]55%    rasterMath... [########################................]61%    rasterMath... [##########################..............]66%    rasterMath... [############################............]72%    rasterMath... [###############################.........]77%    rasterMath... [#################################.......]83%    rasterMath... [###################################.....]88%    rasterMath... [#####################################...]94%    rasterMath... [########################################]100%
    Saved /tmp/modal.tif using function modalClass


Plot result


.. code-block:: default


    import gdal
    from matplotlib import pyplot as plt 

    src = gdal.Open('/tmp/modal.tif')
    plt.imshow(src.ReadAsArray()[0,:,:])



.. image:: /auto_examples/raster_tools/images/sphx_glr_modalClass_001.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.332 seconds)


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
