.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_learnTools_learnWithRFandRS50vector.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_learnTools_learnWithRFandRS50vector.py:


Learn from vector with Random-Forest and Random Sampling 50% (RS50)
====================================================================

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


    X,y = datasets.getHistoricalMap(return_X_y=True)







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
    LAP.learnFromVector(X,y,cv=RS50,
                        classifier=classifier,param_grid=dict(n_estimators=[100,200]))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Fitting 10 folds for each of 2 candidates, totalling 20 fits
    best n_estimators : 200


Get kappa from each fold
---------------------------



.. code-block:: python

  
    for kappa in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
        print(kappa)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [0.94256008314765816]
    [0.94227598585009031]
    [0.94256008314765816]
    [0.94227598585009031]
    [0.94256008314765816]
    [0.94227598585009031]
    [0.94256008314765816]
    [0.94227598585009031]
    [0.94256008314765816]
    [0.94227598585009031]


Get each confusion matrix from folds
-----------------------------------------------



.. code-block:: python


    for cm in LAP.getStatsFromCV(confusionMatrix=True):
        print(cm)
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [array([[3677,   80,    2,   12,    0],
           [  67, 1068,    1,   11,    0],
           [   0,    0, 1140,    0,    0],
           [  10,   20,    3,  230,    0],
           [   3,    0,    1,    0,    0]])]
    [array([[3693,   68,    1,    9,    0],
           [  82, 1050,    0,   14,    0],
           [   2,    0, 1137,    0,    0],
           [  12,   17,    1,  232,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3677,   80,    2,   12,    0],
           [  67, 1068,    1,   11,    0],
           [   0,    0, 1140,    0,    0],
           [  10,   20,    3,  230,    0],
           [   3,    0,    1,    0,    0]])]
    [array([[3693,   68,    1,    9,    0],
           [  82, 1050,    0,   14,    0],
           [   2,    0, 1137,    0,    0],
           [  12,   17,    1,  232,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3677,   80,    2,   12,    0],
           [  67, 1068,    1,   11,    0],
           [   0,    0, 1140,    0,    0],
           [  10,   20,    3,  230,    0],
           [   3,    0,    1,    0,    0]])]
    [array([[3693,   68,    1,    9,    0],
           [  82, 1050,    0,   14,    0],
           [   2,    0, 1137,    0,    0],
           [  12,   17,    1,  232,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3677,   80,    2,   12,    0],
           [  67, 1068,    1,   11,    0],
           [   0,    0, 1140,    0,    0],
           [  10,   20,    3,  230,    0],
           [   3,    0,    1,    0,    0]])]
    [array([[3693,   68,    1,    9,    0],
           [  82, 1050,    0,   14,    0],
           [   2,    0, 1137,    0,    0],
           [  12,   17,    1,  232,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3677,   80,    2,   12,    0],
           [  67, 1068,    1,   11,    0],
           [   0,    0, 1140,    0,    0],
           [  10,   20,    3,  230,    0],
           [   3,    0,    1,    0,    0]])]
    [array([[3693,   68,    1,    9,    0],
           [  82, 1050,    0,   14,    0],
           [   2,    0, 1137,    0,    0],
           [  12,   17,    1,  232,    0],
           [   4,    0,    0,    0,    0]])]


Save each confusion matrix from folds
-----------------------------------------------



.. code-block:: python


    LAP.saveCMFromCV('/tmp/testMTB/',prefix='RS50_',header=False)







Predict map
---------------------------



.. code-block:: python

    raster,_ = datasets.getHistoricalMap()
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



.. image:: /auto_examples/learnTools/images/sphx_glr_learnWithRFandRS50vector_001.png
    :class: sphx-glr-single-img




**Total running time of the script:** ( 0 minutes  35.037 seconds)


.. _sphx_glr_download_auto_examples_learnTools_learnWithRFandRS50vector.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: learnWithRFandRS50vector.py <learnWithRFandRS50vector.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: learnWithRFandRS50vector.ipynb <learnWithRFandRS50vector.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
