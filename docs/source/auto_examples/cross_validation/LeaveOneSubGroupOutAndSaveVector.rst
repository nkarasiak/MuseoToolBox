.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_cross_validation_LeaveOneSubGroupOutAndSaveVector.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_cross_validation_LeaveOneSubGroupOutAndSaveVector.py:


Generate a cross-validation and/or save each fold to a vector file
===================================================================

This example shows how to make a Leave-One-SubGroup-Out and save
each fold as a vector file.



Import librairies
-------------------------------------------



.. code-block:: python


    from museotoolbox.cross_validation import LeaveOneSubGroupOut
    from museotoolbox.raster_tools import getSamplesFromROI
    from museotoolbox import datasets







Load HistoricalMap dataset
-------------------------------------------



.. code-block:: python


    raster,vector = datasets.historicalMap()
    field = 'Class'
    group = 'uniquefid'
    X,y,s = getSamplesFromROI(raster,vector,field,group)






Create CV
-------------------------------------------



.. code-block:: python


    valid_size = 0.5 # Means 50%
    LOSGO = LeaveOneSubGroupOut(verbose=False,random_state=12)







.. note::
   Split is made to generate each fold



.. code-block:: python


    LOSGO.get_n_splits(X,y,s)
    for tr,vl in LOSGO.split(X,y,s):
        print(tr.shape,vl.shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (9308,) (3339,)
    (10873,) (1774,)


Save each train/valid fold to a vector file (here in polygon type)




.. code-block:: python


    vectorFiles = LOSGO.saveVectorFiles(vector,field,groupsField=group,outVector='/tmp/LOSGO.gpkg')

    for tr,vl in vectorFiles:
        print(tr,vl)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Warning : This function generates vector files according to your vector.
            The number of features may differ from the number of pixels used in classification.
            If you want to save every ROI pixels in the vector, please use vector_tools.sampleExtraction before.
    /tmp/LOSGO_train_0.gpkg /tmp/LOSGO_valid_0.gpkg
    /tmp/LOSGO_train_1.gpkg /tmp/LOSGO_valid_1.gpkg


The sampling can be different in vector point or polygon.
So you can generate each centroid of a pixel that contains the polygon.




.. code-block:: python

    
    from museotoolbox.vector_tools import sampleExtraction
    vectorPointPerPixel = '/tmp/vectorCentroid.gpkg'
    sampleExtraction(raster,vector,vectorPointPerPixel)

    vectorFiles = LOSGO.saveVectorFiles(vectorPointPerPixel,field,groupsField=group,outVector='/tmp/LOSGO.gpkg')

    for tr,vl in LOSGO.split(X,y,s):
        print(tr.shape,vl.shape)



.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Adding 'uniquefid' field to the original vector.
    Field 'uniquefid' is already in /mnt/DATA/lib/MuseoToolBox/museotoolbox/datasets/historicalmap/train.gpkg
    Extract values from raster...
    Reading raster values...  [........................................]0%    Reading raster values...  [##################......................]45%    Reading raster values...  [####################################....]90%    Reading raster values...  [########################################]100%
    Adding each centroid to /tmp/vectorCentroid.gpkg...
    Adding points...  [........................................]0%    Adding points...  [........................................]1%    Adding points...  [........................................]2%    Adding points...  [#.......................................]3%    Adding points...  [#.......................................]4%    Adding points...  [##......................................]5%    Adding points...  [##......................................]6%    Adding points...  [##......................................]7%    Adding points...  [###.....................................]8%    Adding points...  [###.....................................]9%    Adding points...  [####....................................]10%    Adding points...  [####....................................]11%    Adding points...  [####....................................]12%    Adding points...  [#####...................................]13%    Adding points...  [#####...................................]14%    Adding points...  [######..................................]15%    Adding points...  [######..................................]16%    Adding points...  [######..................................]17%    Adding points...  [#######.................................]18%    Adding points...  [#######.................................]19%    Adding points...  [########................................]20%    Adding points...  [########................................]21%    Adding points...  [########................................]22%    Adding points...  [#########...............................]23%    Adding points...  [#########...............................]24%    Adding points...  [##########..............................]25%    Adding points...  [##########..............................]26%    Adding points...  [##########..............................]27%    Adding points...  [###########.............................]28%    Adding points...  [###########.............................]29%    Adding points...  [############............................]30%    Adding points...  [############............................]31%    Adding points...  [############............................]32%    Adding points...  [#############...........................]33%    Adding points...  [#############...........................]34%    Adding points...  [##############..........................]35%    Adding points...  [##############..........................]36%    Adding points...  [##############..........................]37%    Adding points...  [###############.........................]38%    Adding points...  [###############.........................]39%    Adding points...  [################........................]40%    Adding points...  [################........................]41%    Adding points...  [################........................]42%    Adding points...  [#################.......................]43%    Adding points...  [#################.......................]44%    Adding points...  [##################......................]45%    Adding points...  [##################......................]46%    Adding points...  [##################......................]47%    Adding points...  [###################.....................]48%    Adding points...  [###################.....................]49%    Adding points...  [####################....................]50%    Adding points...  [####################....................]51%    Adding points...  [####################....................]52%    Adding points...  [#####################...................]53%    Adding points...  [#####################...................]54%    Adding points...  [######################..................]55%    Adding points...  [######################..................]56%    Adding points...  [######################..................]57%    Adding points...  [#######################.................]58%    Adding points...  [#######################.................]59%    Adding points...  [########################................]60%    Adding points...  [########################................]61%    Adding points...  [########################................]62%    Adding points...  [#########################...............]63%    Adding points...  [#########################...............]64%    Adding points...  [##########################..............]65%    Adding points...  [##########################..............]66%    Adding points...  [##########################..............]67%    Adding points...  [###########################.............]68%    Adding points...  [###########################.............]69%    Adding points...  [############################............]70%    Adding points...  [############################............]71%    Adding points...  [############################............]72%    Adding points...  [#############################...........]73%    Adding points...  [#############################...........]74%    Adding points...  [##############################..........]75%    Adding points...  [##############################..........]76%    Adding points...  [##############################..........]77%    Adding points...  [###############################.........]78%    Adding points...  [###############################.........]79%    Adding points...  [################################........]80%    Adding points...  [################################........]81%    Adding points...  [################################........]82%    Adding points...  [#################################.......]83%    Adding points...  [#################################.......]84%    Adding points...  [##################################......]85%    Adding points...  [##################################......]86%    Adding points...  [##################################......]87%    Adding points...  [###################################.....]88%    Adding points...  [###################################.....]89%    Adding points...  [####################################....]90%    Adding points...  [####################################....]91%    Adding points...  [####################################....]92%    Adding points...  [#####################################...]93%    Adding points...  [#####################################...]94%    Adding points...  [######################################..]95%    Adding points...  [######################################..]96%    Adding points...  [######################################..]97%    Adding points...  [#######################################.]98%    Adding points...  [#######################################.]99%    Adding points...  [########################################]100%
    (9308,) (3339,)
    (10873,) (1774,)


**Total running time of the script:** ( 0 minutes  2.968 seconds)


.. _sphx_glr_download_auto_examples_cross_validation_LeaveOneSubGroupOutAndSaveVector.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: LeaveOneSubGroupOutAndSaveVector.py <LeaveOneSubGroupOutAndSaveVector.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: LeaveOneSubGroupOutAndSaveVector.ipynb <LeaveOneSubGroupOutAndSaveVector.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
