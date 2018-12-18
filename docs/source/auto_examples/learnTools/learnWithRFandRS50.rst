.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_learnTools_learnWithRFandRS50.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_learnTools_learnWithRFandRS50.py:


Learn with Random-Forest and Random Sampling 50% (RS50)
========================================================

This example shows how to make a Random Sampling with 
50% for each class.



Import librairies
-------------------------------------------



.. code-block:: python


    from museotoolbox.learnTools import learnAndPredict
    from museotoolbox.crossValidation import RandomCV
    from museotoolbox import datasets
    from sklearn.ensemble import RandomForestClassifier







Load HistoricalMap dataset
-------------------------------------------



.. code-block:: python


    raster,vector = datasets.getHistoricalMap()
    field = 'Class'






Create CV
-------------------------------------------



.. code-block:: python

    RS50 = RandomCV(valid_size=0.5,n_splits=10,
                    random_state=12,verbose=False)






Initialize Random-Forest
---------------------------



.. code-block:: python


    classifier = RandomForestClassifier(random_state=12)







Start learning
---------------------------



.. code-block:: python



    LAP = learnAndPredict(n_jobs=-1)
    LAP.learnFromRaster(raster,vector,field,cv=RS50,
                        classifier=classifier,param_grid=dict(n_estimators=[100,200]))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Fitting 10 folds for each of 2 candidates, totalling 20 fits
    best n_estimators : 200


Get kappa from each fold
---------------------------



.. code-block:: python

  
    for stats in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
        print(stats['kappa'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

<<<<<<< HEAD
    0.942560083148
    0.94227598585
    0.942560083148
    0.94227598585
    0.942560083148
    0.94227598585
    0.942560083148
    0.94227598585
    0.942560083148
    0.94227598585
=======
    [0.94252747485260613]
    [0.94284524103840484]
    [0.94280762977390864]
    [0.94175058724703875]
    [0.94416532636909412]
    [0.94117648081866345]
    [0.94307488396034034]
    [0.94147139724909235]
    [0.94113676225896314]
    [0.94256617983102708]
>>>>>>> master


Get each confusion matrix from folds
-----------------------------------------------



.. code-block:: python


<<<<<<< HEAD
    for stats in LAP.getStatsFromCV(confusionMatrix=True):
        print(stats['confusionMatrix'])
=======
    for cm in LAP.getStatsFromCV(confusionMatrix=True):
        print(cm)
>>>>>>> master
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

<<<<<<< HEAD
    [[3677   80    2   12    0]
     [  67 1068    1   11    0]
     [   0    0 1140    0    0]
     [  10   20    3  230    0]
     [   3    0    1    0    0]]
    [[3693   68    1    9    0]
     [  82 1050    0   14    0]
     [   2    0 1137    0    0]
     [  12   17    1  232    0]
     [   4    0    0    0    0]]
    [[3677   80    2   12    0]
     [  67 1068    1   11    0]
     [   0    0 1140    0    0]
     [  10   20    3  230    0]
     [   3    0    1    0    0]]
    [[3693   68    1    9    0]
     [  82 1050    0   14    0]
     [   2    0 1137    0    0]
     [  12   17    1  232    0]
     [   4    0    0    0    0]]
    [[3677   80    2   12    0]
     [  67 1068    1   11    0]
     [   0    0 1140    0    0]
     [  10   20    3  230    0]
     [   3    0    1    0    0]]
    [[3693   68    1    9    0]
     [  82 1050    0   14    0]
     [   2    0 1137    0    0]
     [  12   17    1  232    0]
     [   4    0    0    0    0]]
    [[3677   80    2   12    0]
     [  67 1068    1   11    0]
     [   0    0 1140    0    0]
     [  10   20    3  230    0]
     [   3    0    1    0    0]]
    [[3693   68    1    9    0]
     [  82 1050    0   14    0]
     [   2    0 1137    0    0]
     [  12   17    1  232    0]
     [   4    0    0    0    0]]
    [[3677   80    2   12    0]
     [  67 1068    1   11    0]
     [   0    0 1140    0    0]
     [  10   20    3  230    0]
     [   3    0    1    0    0]]
    [[3693   68    1    9    0]
     [  82 1050    0   14    0]
     [   2    0 1137    0    0]
     [  12   17    1  232    0]
     [   4    0    0    0    0]]


Save each confusion matrix from folds
-----------------------------------------------



.. code-block:: python


    LAP.saveCMFromCV('/tmp/testMTB/',prefix='RS50_')





=======
    [array([[3679,   80,    2,   10,    0],
           [  66, 1069,    1,   11,    0],
           [   0,    0, 1140,    0,    0],
           [   9,   19,    3,  232,    0],
           [   3,    0,    1,    0,    0]])]
    [array([[3691,   69,    1,   10,    0],
           [  80, 1052,    0,   14,    0],
           [   2,    0, 1137,    0,    0],
           [  11,   17,    1,  233,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3681,   78,    2,   10,    0],
           [  68, 1066,    1,   12,    0],
           [   0,    0, 1140,    0,    0],
           [   9,   20,    3,  231,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3692,   68,    1,   10,    0],
           [  85, 1047,    0,   14,    0],
           [   2,    0, 1137,    0,    0],
           [  13,   18,    1,  230,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3677,   80,    2,   12,    0],
           [  71, 1064,    1,   11,    0],
           [   0,    0, 1140,    0,    0],
           [   9,   20,    3,  231,    0],
           [   3,    1,    0,    0,    0]])]
    [array([[3695,   65,    1,   10,    0],
           [  80, 1053,    0,   13,    0],
           [   2,    0, 1137,    0,    0],
           [  10,   19,    1,  232,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3680,   78,    2,   11,    0],
           [  72, 1063,    1,   11,    0],
           [   0,    0, 1140,    0,    0],
           [   9,   20,    3,  231,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3688,   71,    1,   11,    0],
           [  80, 1054,    0,   12,    0],
           [   2,    0, 1137,    0,    0],
           [  12,   16,    1,  233,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3679,   78,    2,   12,    0],
           [  71, 1064,    1,   11,    0],
           [   0,    0, 1140,    0,    0],
           [  10,   21,    3,  229,    0],
           [   3,    0,    1,    0,    0]])]
    [array([[3690,   69,    1,   11,    0],
           [  78, 1056,    0,   12,    0],
           [   2,    0, 1137,    0,    0],
           [  10,   19,    1,  232,    0],
           [   4,    0,    0,    0,    0]])]
>>>>>>> master


Save each confusion matrix from folds
-----------------------------------------------



.. code-block:: python


    LAP.saveCMFromCV('/tmp/testMTB/',prefix='RS50_')







Predict map
---------------------------



.. code-block:: python

    
    LAP.predictRaster(raster,'/tmp/classification.tif')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Prediction...  [##################......................]45%    Prediction...  [####################################....]90%    Saved /tmp/classification.tif using function predictArray


Plot example



.. code-block:: python


    from matplotlib import pyplot as plt
    import gdal
    src=gdal.Open('/tmp/classification.tif')
    plt.imshow(src.GetRasterBand(1).ReadAsArray(),cmap=plt.get_cmap('tab20'))
    plt.axis('off')
    plt.show()



.. image:: /auto_examples/learnTools/images/sphx_glr_learnWithRFandRS50_001.png
    :class: sphx-glr-single-img




<<<<<<< HEAD
**Total running time of the script:** ( 0 minutes  30.839 seconds)
=======
**Total running time of the script:** ( 0 minutes  39.676 seconds)
>>>>>>> master


.. _sphx_glr_download_auto_examples_learnTools_learnWithRFandRS50.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: learnWithRFandRS50.py <learnWithRFandRS50.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: learnWithRFandRS50.ipynb <learnWithRFandRS50.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
