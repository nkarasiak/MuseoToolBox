.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_learn_tools_learnWithRFandRS50.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_learn_tools_learnWithRFandRS50.py:


Learn with Random-Forest and Random Sampling 50% (RS50)
========================================================

This example shows how to make a Random Sampling with 
50% for each class.



Import librairies
-------------------------------------------



.. code-block:: python


    from museotoolbox.learn_tools import learnAndPredict
    from museotoolbox.cross_validation import RandomCV
    from museotoolbox import datasets
    from museotoolbox import raster_tools
    from sklearn.ensemble import RandomForestClassifier







Load HistoricalMap dataset
-------------------------------------------



.. code-block:: python


    raster,vector = datasets.historicalMap()
    field = 'Class'







Create CV
-------------------------------------------



.. code-block:: python


    RS50 = RandomCV(valid_size=0.5,n_splits=2,
                    random_state=12,verbose=False)







Initialize Random-Forest
---------------------------



.. code-block:: python


    classifier = RandomForestClassifier(random_state=12,n_jobs=-1)







Start learning
---------------------------



.. code-block:: python


    LAP = learnAndPredict(n_jobs=-1,verbose=1)
    LAP.learnFromRaster(raster,vector,field,cv=RS50,
                        classifier=classifier,param_grid=dict(n_estimators=[100,200]))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Reading raster values...  [........................................]0%    Reading raster values...  [##################......................]45%    Reading raster values...  [####################################....]90%    Reading raster values...  [########################################]100%
    Fitting 2 folds for each of 2 candidates, totalling 4 fits
    best score : 0.966624485922
    best n_estimators : 200


Read the model
-------------------



.. code-block:: python

    print(LAP.model)
    print(LAP.model.cv_results_)
    print(LAP.model.best_score_)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    GridSearchCV(cv=<museotoolbox.cross_validation.RandomCV object at 0x7fd9f4a4dc88>,
           error_score='raise',
           estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1,
                oob_score=False, random_state=12, verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=-1,
           param_grid={'n_estimators': [100, 200]}, pre_dispatch='2*n_jobs',
           refit=True, return_train_score='warn', scoring='accuracy',
           verbose=1)
    {'mean_fit_time': array([ 0.63643289,  1.06290293]), 'std_fit_time': array([ 0.00264573,  0.02864861]), 'mean_score_time': array([ 0.10340083,  0.20547557]), 'std_score_time': array([ 0.00057447,  0.00018334]), 'param_n_estimators': masked_array(data = [100 200],
                 mask = [False False],
           fill_value = ?)
    , 'params': [{'n_estimators': 100}, {'n_estimators': 200}], 'split0_test_score': array([ 0.96630813,  0.96678266]), 'split1_test_score': array([ 0.96646631,  0.96646631]), 'mean_test_score': array([ 0.96638722,  0.96662449]), 'std_test_score': array([  7.90888959e-05,   1.58177792e-04]), 'rank_test_score': array([2, 1], dtype=int32), 'split0_train_score': array([ 0.99794466,  0.99794466]), 'split1_train_score': array([ 0.99794466,  0.99794466]), 'mean_train_score': array([ 0.99794466,  0.99794466]), 'std_train_score': array([ 0.,  0.])}
    0.966624485922


Get kappa from each fold
---------------------------



.. code-block:: python

  
    for stats in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
        print(stats['kappa'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.94227598585
    0.94193832769


Get each confusion matrix from folds
-----------------------------------------------



.. code-block:: python


    for stats in LAP.getStatsFromCV(confusionMatrix=True):
        print(stats['confusionMatrix'])
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[3693   68    1    9    0]
     [  82 1050    0   14    0]
     [   2    0 1137    0    0]
     [  12   17    1  232    0]
     [   4    0    0    0    0]]
    [[3679   79    2   11    0]
     [  70 1063    1   12    0]
     [   0    0 1139    0    0]
     [   8   22    3  229    0]
     [   3    1    0    0    0]]


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



.. image:: /auto_examples/learn_tools/images/sphx_glr_learnWithRFandRS50_001.png
    :class: sphx-glr-single-img




**Total running time of the script:** ( 0 minutes  13.689 seconds)


.. _sphx_glr_download_auto_examples_learn_tools_learnWithRFandRS50.py:


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
