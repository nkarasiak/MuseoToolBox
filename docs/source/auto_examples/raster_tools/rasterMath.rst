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
    [[166 153 136]
     [193 180 161]
     [170 153 133]
     [196 180 155]
     [186 166 141]
     [159 140 110]
     [211 187 159]
     [185 162 130]
     [185 162 130]
     [189 169 134]
     [199 179 146]
     [182 166 132]
     [169 152 122]
     [172 159 127]
     [109 95 66]
     [66 57 28]
     [62 56 32]
     [65 60 38]
     [135 128 109]
     [183 175 156]
     [184 172 156]
     [121 105 90]
     [195 177 163]
     [172 153 139]
     [180 157 143]
     [180 157 141]
     [183 161 140]
     [173 151 127]
     [177 157 130]
     [142 123 91]
     [177 158 125]
     [127 107 72]
     [128 99 67]
     [173 144 112]
     [178 148 120]
     [164 139 109]
     [180 159 132]
     [147 130 102]
     [166 150 125]
     [173 159 133]
     [173 159 132]
     [111 95 69]
     [112 95 67]
     [132 111 82]
     [154 129 99]
     [137 108 76]
     [131 100 69]
     [105 59 35]
     [165 87 85]
     [156 67 73]
     [181 166 133]
     [203 187 154]
     [201 181 148]
     [211 188 156]
     [203 180 149]
     [208 185 154]
     [194 171 140]
     [202 179 148]
     [191 167 139]
     [190 166 138]
     [196 172 144]
     [181 157 129]
     [193 169 143]
     [173 149 123]
     [169 145 119]
     [190 166 140]
     [183 159 135]
     [181 157 133]
     [197 175 152]
     [155 133 110]
     [203 181 158]
     [169 147 124]
     [178 156 133]
     [191 169 146]
     [207 187 162]
     [199 179 154]
     [190 173 147]
     [166 149 123]
     [170 153 125]
     [128 111 83]
     [184 169 140]
     [199 184 155]
     [168 153 122]
     [200 185 156]
     [158 140 120]
     [128 109 92]
     [113 89 77]
     [204 180 168]
     [101 74 65]
     [183 159 149]
     [88 70 60]
     [169 157 145]
     [83 75 62]
     [187 185 170]
     [60 59 41]
     [75 74 56]
     [73 68 49]
     [170 162 143]
     [103 90 73]
     [180 165 146]]


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




**Total running time of the script:** ( 0 minutes  0.830 seconds)


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
