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
    [[134 120  92]
     [106  89  63]
     [137 133  83]
     [134 126  85]
     [120 108  79]
     [127 124  87]
     [111 103  84]
     [102  89  78]
     [118 117  77]
     [ 98  91  54]
     [116 110  76]
     [127 123  88]
     [112 104  73]
     [161 151 107]
     [120 108  86]
     [116 110  64]
     [145 142  87]
     [118 108  94]
     [123 114  98]
     [113 110  68]
     [119 113  86]
     [142 135 106]
     [139 134 102]
     [128 122  88]
     [147 137 103]
     [126 118  79]
     [125 118  84]
     [131 126  91]
     [134 125  93]
     [131 113  94]
     [117 122  57]
     [104  96  60]
     [137 120 104]
     [119 125  71]
     [119 116  87]
     [ 71  52  51]
     [116 121  66]
     [ 84  78  40]
     [ 84  79  43]
     [100  94  61]
     [141 132 103]
     [129 121  88]
     [153 149 110]
     [124 118  79]
     [154 144 108]
     [116 102  72]
     [113 109  67]
     [139 124 103]
     [141 138  97]
     [138 141  91]
     [117 106  86]
     [112 106  83]
     [ 69  74  29]
     [134 126  94]
     [127 122  78]
     [117 113  74]
     [172 167 135]
     [114 106  80]
     [148 141 111]
     [124 121  82]
     [131 126  84]
     [119 109  68]
     [118 113  76]
     [159 155 109]
     [143 137  99]
     [139 130 101]
     [130 112  88]
     [ 92 102  65]
     [ 92  93  65]
     [104  86  65]
     [133 138  94]
     [116 111  79]
     [148 144 109]
     [135 130  94]
     [140 132  96]
     [ 93  86  46]
     [158 154 112]
     [125 120  82]
     [138 128  99]
     [151 146 108]
     [168 161 118]
     [102  93  63]
     [ 82  79  54]
     [ 92  95  75]
     [117 108  69]
     [115 110  76]
     [115 110  77]
     [126 120  89]
     [ 89  80  50]
     [103  95  62]
     [ 88  83  45]
     [118 112  76]
     [113 103  72]
     [131 115  75]
     [105  91  50]
     [119 118  91]
     [106  99  61]
     [113 108  74]
     [110 104  79]
     [102  93  74]
     [120 111  89]
     [ 94  89  58]
     [140 133  97]
     [133 121  86]
     [ 99  99  64]
     [ 93  84  48]
     [ 97  90  63]
     [106  96  70]
     [111 102  72]
     [ 96  91  57]
     [138 131 102]
     [118 106  83]
     [103  91  65]
     [141 135  85]
     [156 136  85]
     [143 133  84]
     [142 132  87]
     [140 126  87]
     [155 146 103]
     [162 156 110]]


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

   **Total running time of the script:** ( 0 minutes  0.361 seconds)


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
