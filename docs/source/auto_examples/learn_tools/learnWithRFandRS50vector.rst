.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_learn_tools_learnWithRFandRS50vector.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_learn_tools_learnWithRFandRS50vector.py:


Learn from vector with Random-Forest and Random Sampling 50% (RS50)
====================================================================

This example shows how to make a Random Sampling with 
50% for each class.


Import librairies
-------------------------------------------


.. code-block:: default


    from museotoolbox.learn_tools import learnAndPredict
    from museotoolbox.cross_validation import RandomStratifiedKFold
    from museotoolbox import datasets
    from sklearn.ensemble import RandomForestClassifier







Load HistoricalMap dataset
-------------------------------------------


.. code-block:: default


    X,y = datasets.historicalMap(return_X_y=True,low_res=True)







Create CV
-------------------------------------------


.. code-block:: default

    SKF = RandomStratifiedKFold(n_splits=2,n_repeats=5,
                    random_state=12,verbose=False)







Initialize Random-Forest
---------------------------


.. code-block:: default


    classifier = RandomForestClassifier(random_state=12)







Start learning
---------------------------


.. code-block:: default


    LAP = learnAndPredict(n_jobs=1)
    LAP.learnFromVector(X,y,cv=SKF,
                        classifier=classifier,param_grid=dict(n_estimators=[10]))







Get kappa from each fold
---------------------------


.. code-block:: default

  
    for stats in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
        print(stats['kappa'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.9034198483801366
    0.8769288132308425
    0.8933293550459626
    0.8892646474406427
    0.9058717010561249
    0.8853759286419574
    0.8963182153236124
    0.8943277744550842
    0.8963299999647014
    0.8977575609560402


Get each confusion matrix from folds
-----------------------------------------------


.. code-block:: default


    for stats in LAP.getStatsFromCV(confusionMatrix=True):
        print(stats['confusionMatrix'])
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[925  16   0   1   0]
     [ 37 238   0  11   0]
     [  0   0 284   0   0]
     [  1  19   1  45   0]
     [  1   0   0   0   0]]
    [[913  25   0   4   0]
     [ 48 227   0  11   0]
     [  0   0 283   1   0]
     [  2  18   1  45   0]
     [  0   0   0   1   0]]
    [[917  25   0   0   0]
     [ 42 240   0   4   0]
     [  0   0 283   1   0]
     [  3  19   1  43   0]
     [  1   0   0   0   0]]
    [[925  13   0   4   0]
     [ 54 224   0   8   0]
     [  1   0 282   1   0]
     [  2  13   2  49   0]
     [  0   1   0   0   0]]
    [[919  22   0   1   0]
     [ 41 240   0   5   0]
     [  0   0 282   2   0]
     [  0  12   1  53   0]
     [  0   1   0   0   0]]
    [[919  23   0   0   0]
     [ 47 228   0  11   0]
     [  0   0 284   0   0]
     [  2  18   1  45   0]
     [  1   0   0   0   0]]
    [[927  15   0   0   0]
     [ 46 226   0  13   1]
     [  0   0 283   1   0]
     [  2  13   1  50   0]
     [  1   0   0   0   0]]
    [[919  20   0   3   0]
     [ 48 233   0   5   0]
     [  0   0 284   0   0]
     [  0  16   2  48   0]
     [  1   0   0   0   0]]
    [[924  17   0   1   0]
     [ 52 223   0  11   0]
     [  0   0 283   1   0]
     [  1   7   2  56   0]
     [  0   1   0   0   0]]
    [[921  19   0   2   0]
     [ 39 246   0   1   0]
     [  0   0 280   4   0]
     [  2  24   0  40   0]
     [  1   0   0   0   0]]


Only get accuracies score (OA and Kappa)
-----------------------------------------------


.. code-block:: default


    for stats in LAP.getStatsFromCV(OA=True,kappa=True,confusionMatrix=False,F1=False):
        print(stats)
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    {'kappa': 0.9034198483801366, 'OA': 0.9449018366054465}
    {'kappa': 0.8769288132308425, 'OA': 0.9297023432552248}
    {'kappa': 0.8933293550459626, 'OA': 0.9392020265991133}
    {'kappa': 0.8892646474406427, 'OA': 0.9373020899303357}
    {'kappa': 0.9058717010561249, 'OA': 0.9461684610512983}
    {'kappa': 0.8853759286419574, 'OA': 0.934768841038632}
    {'kappa': 0.8963182153236124, 'OA': 0.9411019632678911}
    {'kappa': 0.8943277744550842, 'OA': 0.9398353388220393}
    {'kappa': 0.8963299999647014, 'OA': 0.9411019632678911}
    {'kappa': 0.8977575609560402, 'OA': 0.9417352754908169}


Save each confusion matrix from folds
-----------------------------------------------


.. code-block:: default


    LAP.saveCMFromCV('/tmp/testMTB/',prefix='SKF_',header=True)
  






Predict map
---------------------------


.. code-block:: default

    raster,_ = datasets.historicalMap(low_res=True)
    LAP.predictRaster(raster,'/tmp/classification.tif')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 18
    Detected 1 band for function predictArray.
    Prediction... [........................................]0%    Prediction... [##......................................]5%    Prediction... [####....................................]11%    Prediction... [######..................................]16%    Prediction... [########................................]22%    Prediction... [###########.............................]27%    Prediction... [#############...........................]33%    Prediction... [###############.........................]38%    Prediction... [#################.......................]44%    Prediction... [####################....................]50%    Prediction... [######################..................]55%    Prediction... [########################................]61%    Prediction... [##########################..............]66%    Prediction... [############################............]72%    Prediction... [###############################.........]77%    Prediction... [#################################.......]83%    Prediction... [###################################.....]88%    Prediction... [#####################################...]94%    Prediction... [########################################]100%
    Saved /tmp/classification.tif using function predictArray


Plot example


.. code-block:: default


    from matplotlib import pyplot as plt
    import gdal
    src=gdal.Open('/tmp/classification.tif')
    plt.imshow(src.GetRasterBand(1).ReadAsArray(),cmap=plt.get_cmap('tab20'))
    plt.axis('off')
    plt.show()



.. image:: /auto_examples/learn_tools/images/sphx_glr_learnWithRFandRS50vector_001.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  1.645 seconds)


.. _sphx_glr_download_auto_examples_learn_tools_learnWithRFandRS50vector.py:


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
