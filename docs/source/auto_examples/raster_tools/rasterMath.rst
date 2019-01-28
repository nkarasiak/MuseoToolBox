.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_raster_tools_rasterMath.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_raster_tools_rasterMath.py:


Basics to use rasterMath
===============================================================

Compute substract and addition between two raster bands.



Import librairies
-------------------------------------------



.. code-block:: python


    from museotoolbox.raster_tools import rasterMath
    from museotoolbox import datasets
    import numpy as np






Load HistoricalMap dataset
-------------------------------------------



.. code-block:: python


    raster,vector = datasets.historicalMap()







Initialize rasterMath with raster
------------------------------------



.. code-block:: python


    rM = rasterMath(raster)

    print(rM.getRandomBlock())





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 15
    [[160 151 134]
     [113 104 87]
     [154 143 125]
     [187 176 156]
     [162 149 130]
     [189 177 155]
     [164 149 128]
     [173 158 137]
     [193 181 159]
     [141 129 107]
     [63 52 34]
     [92 84 65]
     [106 97 80]
     [73 66 48]
     [62 56 42]
     [103 96 80]
     [181 169 155]
     [157 144 128]
     [139 123 107]
     [188 173 154]
     [152 135 117]
     [190 173 153]
     [121 103 83]
     [182 165 145]
     [166 149 133]
     [109 96 80]
     [131 119 105]
     [119 111 100]
     [133 129 120]
     [106 106 98]
     [55 54 50]
     [69 67 55]
     [187 179 158]
     [190 179 151]
     [173 162 134]
     [177 166 138]
     [188 177 149]
     [172 161 133]
     [194 183 155]
     [140 129 101]
     [202 191 163]
     [194 183 155]
     [202 191 163]
     [185 174 146]
     [195 184 156]
     [180 169 141]
     [150 139 111]
     [115 105 80]
     [100 91 74]
     [117 109 96]
     [140 125 118]
     [110 97 88]
     [101 89 73]
     [80 72 51]
     [70 58 36]
     [151 137 111]
     [144 129 100]
     [188 172 139]
     [176 155 124]
     [183 163 130]
     [153 130 98]
     [154 131 100]
     [196 175 146]
     [172 152 127]
     [95 74 55]
     [143 126 110]
     [139 121 107]
     [134 120 107]
     [117 105 93]
     [150 140 128]
     [155 145 133]
     [151 141 129]
     [139 129 117]
     [148 138 126]
     [140 130 118]
     [144 134 122]
     [144 134 122]
     [140 130 118]
     [111 101 89]
     [147 137 125]
     [116 106 94]
     [69 59 47]
     [143 133 121]
     [119 111 100]
     [86 82 71]
     [105 102 93]
     [89 87 75]
     [74 72 60]
     [60 58 45]
     [58 56 43]
     [87 85 70]
     [38 36 21]
     [58 57 39]
     [89 88 70]
     [113 112 92]
     [52 51 31]
     [59 58 38]
     [95 94 74]
     [87 86 65]
     [78 77 56]]


Let's suppose you want compute the difference between blue and green band
I suggest you to define type in numpy array to save space while creating the raster!



.. code-block:: python


    x = rM.getRandomBlock()

    def sub(x):
        return np.array((x[:,0]-x[:,1])).astype(np.int64) 

    def add(x,constant=0):
    
        return np.array((x[:,0]+x[:,1]+constant)).astype(np.int16) 


    rM.addFunction(sub,outRaster='/tmp/sub.tif')
    rM.addFunction(add,outRaster='/tmp/add.tif',constant=10)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Warning : Numpy type int64 is not recognized by gdal. Will use int32 instead
    Using datatype from numpy table : int64
    Detected 1 band(s) for function sub.
    Using datatype from numpy table : int16
    Detected 1 band(s) for function add.


Run the script



.. code-block:: python


    rM.run()





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    rasterMath...  [........................................]0%    rasterMath...  [##......................................]6%    rasterMath...  [#####...................................]13%    rasterMath...  [########................................]20%    rasterMath...  [##########..............................]26%    rasterMath...  [#############...........................]33%    rasterMath...  [################........................]40%    rasterMath...  [##################......................]46%    rasterMath...  [#####################...................]53%    rasterMath...  [########################................]60%    rasterMath...  [##########################..............]66%    rasterMath...  [#############################...........]73%    rasterMath...  [################################........]80%    rasterMath...  [##################################......]86%    rasterMath...  [#####################################...]93%    rasterMath...  [########################################]100%
    Saved /tmp/sub.tif using function sub
    Saved /tmp/add.tif using function add


Plot result



.. code-block:: python


    import gdal
    from matplotlib import pyplot as plt 

    src = gdal.Open('/tmp/add.tif')
    plt.imshow(src.ReadAsArray())



.. image:: /auto_examples/raster_tools/images/sphx_glr_rasterMath_001.png
    :class: sphx-glr-single-img




**Total running time of the script:** ( 0 minutes  1.347 seconds)


.. _sphx_glr_download_auto_examples_raster_tools_rasterMath.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: rasterMath.py <rasterMath.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: rasterMath.ipynb <rasterMath.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
