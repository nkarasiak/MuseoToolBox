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
    [[104 100 71]
     [107 103 76]
     [130 124 102]
     [89 82 63]
     [77 68 53]
     [73 64 47]
     [117 104 88]
     [73 60 41]
     [84 72 48]
     [97 87 60]
     [118 108 83]
     [99 91 68]
     [105 98 80]
     [72 64 51]
     [134 127 121]
     [139 132 126]
     [105 95 86]
     [142 132 123]
     [178 165 159]
     [175 161 158]
     [117 101 101]
     [81 66 63]
     [119 104 99]
     [107 93 84]
     [105 92 76]
     [158 147 129]
     [198 187 167]
     [188 181 162]
     [136 129 113]
     [183 179 150]
     [135 131 104]
     [172 166 144]
     [193 186 167]
     [99 90 75]
     [152 143 126]
     [177 164 148]
     [148 135 116]
     [176 164 140]
     [177 167 140]
     [169 159 134]
     [183 175 152]
     [182 175 157]
     [146 138 125]
     [69 62 56]
     [46 39 33]
     [69 60 51]
     [76 66 56]
     [67 54 48]
     [78 64 61]
     [96 80 80]
     [143 133 108]
     [159 145 118]
     [179 162 134]
     [173 153 128]
     [164 146 122]
     [202 187 166]
     [135 122 106]
     [175 163 149]
     [130 117 108]
     [89 76 67]
     [50 36 25]
     [70 56 43]
     [54 38 23]
     [172 156 140]
     [182 167 148]
     [178 163 144]
     [169 156 137]
     [121 110 92]
     [73 64 49]
     [50 42 31]
     [61 54 46]
     [173 155 135]
     [122 105 87]
     [177 165 149]
     [75 68 52]
     [193 189 177]
     [50 48 33]
     [44 41 24]
     [69 64 42]
     [104 94 69]
     [187 173 146]
     [209 192 164]
     [104 87 61]
     [173 155 131]
     [184 169 148]
     [104 91 75]
     [185 173 161]
     [185 172 163]
     [139 126 117]
     [147 133 124]
     [68 54 41]
     [75 59 46]
     [138 122 106]
     [159 143 127]
     [68 53 34]
     [186 173 156]
     [154 142 126]
     [182 173 158]
     [164 158 146]
     [116 109 101]]


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
    Detected 1 band(s) for output.
    Using datatype from numpy table : int16
    Detected 1 band(s) for output.


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




**Total running time of the script:** ( 0 minutes  1.637 seconds)


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
