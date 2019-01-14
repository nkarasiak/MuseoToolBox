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



.. code-block:: python


    from museotoolbox.learn_tools import learnAndPredict
    from museotoolbox.cross_validation import RandomCV
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
    best n_estimators : 100


Get kappa from each fold
---------------------------



.. code-block:: python

  
    for stats in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
        print(stats['kappa'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.941435720033
    0.941931792221
    0.945978906497
    0.940440589958
    0.946488716767
    0.939721862133
    0.94058560479
    0.938617597316
    0.938303179335
    0.942515715861


Get each confusion matrix from folds
-----------------------------------------------



.. code-block:: python


    for stats in LAP.getStatsFromCV(confusionMatrix=True):
        print(stats['confusionMatrix'])
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[3693   68    1    9    0]
     [  83 1049    0   14    0]
     [   3    0 1136    0    0]
     [  12   18    1  231    0]
     [   4    0    0    0    0]]
    [[3679   78    2   12    0]
     [  70 1062    1   13    0]
     [   0    0 1139    0    0]
     [   9   20    3  230    0]
     [   4    0    0    0    0]]
    [[3692   65    3   11    0]
     [  65 1067    0   14    0]
     [   0    0 1139    0    0]
     [  11   21    3  227    0]
     [   3    0    1    0    0]]
    [[3687   66    1   17    0]
     [  85 1044    1   16    0]
     [   2    0 1137    0    0]
     [   7   17    1  237    0]
     [   4    0    0    0    0]]
    [[3695   63    3   10    0]
     [  66 1065    0   15    0]
     [   1    0 1138    0    0]
     [  13   20    0  229    0]
     [   3    1    0    0    0]]
    [[3699   67    0    5    0]
     [  94 1034    1   17    0]
     [   0    0 1139    0    0]
     [   7   21    3  231    0]
     [   4    0    0    0    0]]
    [[3694   68    2    7    0]
     [  76 1058    0   12    0]
     [   3    0 1136    0    0]
     [  16   28    0  218    0]
     [   4    0    0    0    0]]
    [[3683   69    1   18    0]
     [  83 1038    1   24    0]
     [   2    0 1137    0    0]
     [   3   15    4  240    0]
     [   4    0    0    0    0]]
    [[3675   83    1   12    0]
     [  82 1050    1   13    0]
     [   1    0 1138    0    0]
     [  11   17    0  234    0]
     [   4    0    0    0    0]]
    [[3699   60    3    9    0]
     [  83 1051    0   12    0]
     [   0    0 1139    0    0]
     [   9   28    1  224    0]
     [   4    0    0    0    0]]


Only get accuracies score (OA and Kappa)
-----------------------------------------------



.. code-block:: python


    for stats in LAP.getStatsFromCV(OA=True,kappa=True,confusionMatrix=False,F1=False):
        print(stats)
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    {'kappa': 0.94143572003304099, 'OA': 0.96630813033850049}
    {'kappa': 0.94193179222071344, 'OA': 0.96646630813033851}
    {'kappa': 0.94597890649700211, 'OA': 0.96883897500790894}
    {'kappa': 0.94044058995801239, 'OA': 0.96567541917114841}
    {'kappa': 0.94648871676705537, 'OA': 0.96915533059158498}
    {'kappa': 0.93972186213269504, 'OA': 0.96535906358747237}
    {'kappa': 0.94058560479049313, 'OA': 0.96583359696298643}
    {'kappa': 0.93861759731634042, 'OA': 0.96456817462828215}
    {'kappa': 0.93830317933482466, 'OA': 0.96440999683644413}
    {'kappa': 0.94251571586075289, 'OA': 0.96694084150585258}


Save each confusion matrix from folds
-----------------------------------------------



.. code-block:: python


    LAP.saveCMFromCV('/tmp/testMTB/',prefix='RS50_',header=True)
  






Predict map
---------------------------



.. code-block:: python

    raster,_ = datasets.getHistoricalMap()
    LAP.predictRaster(raster,'/tmp/classification.tif')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Prediction...  [........................................]0%    Prediction...  [##......................................]7%    Prediction...  [#####...................................]14%    Prediction...  [########................................]21%    Prediction...  [###########.............................]28%    Prediction...  [##############..........................]35%    Prediction...  [#################.......................]42%    Prediction...  [####################....................]50%    Prediction...  [######################..................]57%    Prediction...  [#########################...............]64%    Prediction...  [############################............]71%    Prediction...  [###############################.........]78%    Prediction...  [##################################......]85%    Prediction...  [#####################################...]92%    Prediction...  [########################################]100%
    Saved /tmp/classification.tif using function predictArray


Plot example



.. code-block:: python


    from matplotlib import pyplot as plt
    import gdal
    src=gdal.Open('/tmp/classification.tif')
    plt.imshow(src.GetRasterBand(1).ReadAsArray(),cmap=plt.get_cmap('tab20'))
    plt.axis('off')
    plt.show()



.. image:: /auto_examples/learn_tools/images/sphx_glr_learnWithRFandRS50vector_001.png
    :class: sphx-glr-single-img




**Total running time of the script:** ( 0 minutes  29.569 seconds)


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
