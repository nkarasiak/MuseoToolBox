.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_crossValidation_SpatialLeaveAsideOut.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_crossValidation_SpatialLeaveAsideOut.py:


Spatial Leave-Aside-Out (SLAO)
======================================================

This example shows how to make a Spatial Leave-Aside-Out.



Import librairies
-------------------------------------------



.. code-block:: python


    from museotoolbox.crossValidation import SpatialLeaveAsideOut
    from museotoolbox import datasets,rasterTools,vectorTools






Load HistoricalMap dataset
-------------------------------------------



.. code-block:: python


    raster,vector = datasets.getHistoricalMap()
    field = 'Class'
    X,y = rasterTools.getSamplesFromROI(raster,vector,field)
    distanceMatrix = vectorTools.getDistanceMatrix(raster,vector)







Create CV
-------------------------------------------
n_splits will be the number  of the least populated class



.. code-block:: python


    SLOPO = SpatialLeaveAsideOut(valid_size=0.5,n_splits=10,
                                 distanceMatrix=distanceMatrix,random_state=12)

    print(SLOPO.get_n_splits(X,y))






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Warning : n_splits is superior to the number of unique samples/groups
    10


.. note::
   Split is made to generate each fold



.. code-block:: python



    for tr,vl in SLOPO.split(X,y):
        print(tr.shape,vl.shape)
    





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Warning : n_splits is superior to the number of unique samples/groups
    (6222,) (6425,)
    (6243,) (6404,)
    (6296,) (6351,)
    (6307,) (6340,)
    (6288,) (6359,)
    (6277,) (6370,)
    (6277,) (6370,)
    (6274,) (6373,)
    (6246,) (6401,)
    (6273,) (6374,)


Save each train/valid fold in a file
-------------------------------------------
In order to translate polygons into points (each points is a pixel in the raster)
we use sampleExtraction from vectorTools to generate a temporary vector.



.. code-block:: python


    vectorTools.sampleExtraction(raster,vector,outVector='/tmp/pixels.gpkg')

    SLOPO.saveVectorFiles('/tmp/pixels.gpkg',field,outVector='/tmp/SLOPO.gpkg')



.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Adding 'uniquefid' field to the original vector.
    Field 'uniquefid' is already in /mnt/DATA/lib/MuseoToolBox/museotoolbox/datasets/historicalmap/train.gpkg
    Extract values from raster...
    Values from 'uniquefid' field will be extracted
    Reading raster values...  [........................................]0%    Reading raster values...  [##################......................]45%    Reading raster values...  [####################################....]90%    Reading raster values...  [########################################]100%
    Adding each centroid to /tmp/pixels.gpkg...
    Adding points...  [........................................]0%    Adding points...  [........................................]1%    Adding points...  [........................................]2%    Adding points...  [#.......................................]3%    Adding points...  [#.......................................]4%    Adding points...  [##......................................]5%    Adding points...  [##......................................]6%    Adding points...  [##......................................]7%    Adding points...  [###.....................................]8%    Adding points...  [###.....................................]9%    Adding points...  [####....................................]10%    Adding points...  [####....................................]11%    Adding points...  [####....................................]12%    Adding points...  [#####...................................]13%    Adding points...  [#####...................................]14%    Adding points...  [######..................................]15%    Adding points...  [######..................................]16%    Adding points...  [######..................................]17%    Adding points...  [#######.................................]18%    Adding points...  [#######.................................]19%    Adding points...  [########................................]20%    Adding points...  [########................................]21%    Adding points...  [########................................]22%    Adding points...  [#########...............................]23%    Adding points...  [#########...............................]24%    Adding points...  [##########..............................]25%    Adding points...  [##########..............................]26%    Adding points...  [##########..............................]27%    Adding points...  [###########.............................]28%    Adding points...  [###########.............................]29%    Adding points...  [############............................]30%    Adding points...  [############............................]31%    Adding points...  [############............................]32%    Adding points...  [#############...........................]33%    Adding points...  [#############...........................]34%    Adding points...  [##############..........................]35%    Adding points...  [##############..........................]36%    Adding points...  [##############..........................]37%    Adding points...  [###############.........................]38%    Adding points...  [###############.........................]39%    Adding points...  [################........................]40%    Adding points...  [################........................]41%    Adding points...  [################........................]42%    Adding points...  [#################.......................]43%    Adding points...  [#################.......................]44%    Adding points...  [##################......................]45%    Adding points...  [##################......................]46%    Adding points...  [##################......................]47%    Adding points...  [###################.....................]48%    Adding points...  [###################.....................]49%    Adding points...  [####################....................]50%    Adding points...  [####################....................]51%    Adding points...  [####################....................]52%    Adding points...  [#####################...................]53%    Adding points...  [#####################...................]54%    Adding points...  [######################..................]55%    Adding points...  [######################..................]56%    Adding points...  [######################..................]57%    Adding points...  [#######################.................]58%    Adding points...  [#######################.................]59%    Adding points...  [########################................]60%    Adding points...  [########################................]61%    Adding points...  [########################................]62%    Adding points...  [#########################...............]63%    Adding points...  [#########################...............]64%    Adding points...  [##########################..............]65%    Adding points...  [##########################..............]66%    Adding points...  [##########################..............]67%    Adding points...  [###########################.............]68%    Adding points...  [###########################.............]69%    Adding points...  [############################............]70%    Adding points...  [############################............]71%    Adding points...  [############################............]72%    Adding points...  [#############################...........]73%    Adding points...  [#############################...........]74%    Adding points...  [##############################..........]75%    Adding points...  [##############################..........]76%    Adding points...  [##############################..........]77%    Adding points...  [###############################.........]78%    Adding points...  [###############################.........]79%    Adding points...  [################################........]80%    Adding points...  [################################........]81%    Adding points...  [################################........]82%    Adding points...  [#################################.......]83%    Adding points...  [#################################.......]84%    Adding points...  [##################################......]85%    Adding points...  [##################################......]86%    Adding points...  [##################################......]87%    Adding points...  [###################################.....]88%    Adding points...  [###################################.....]89%    Adding points...  [####################################....]90%    Adding points...  [####################################....]91%    Adding points...  [####################################....]92%    Adding points...  [#####################################...]93%    Adding points...  [#####################################...]94%    Adding points...  [######################################..]95%    Adding points...  [######################################..]96%    Adding points...  [######################################..]97%    Adding points...  [#######################################.]98%    Adding points...  [#######################################.]99%    Adding points...  [########################################]100%
    Warning : n_splits is superior to the number of unique samples/groups


**Total running time of the script:** ( 0 minutes  5.901 seconds)


.. _sphx_glr_download_auto_examples_crossValidation_SpatialLeaveAsideOut.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: SpatialLeaveAsideOut.py <SpatialLeaveAsideOut.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: SpatialLeaveAsideOut.ipynb <SpatialLeaveAsideOut.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
