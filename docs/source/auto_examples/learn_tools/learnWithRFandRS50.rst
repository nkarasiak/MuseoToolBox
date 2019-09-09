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


.. code-block:: default


    from museotoolbox.learn_tools import learnAndPredict
    from museotoolbox.cross_validation import RandomStratifiedKFold
    from museotoolbox import datasets
    from museotoolbox import raster_tools
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics







Load HistoricalMap dataset
-------------------------------------------


.. code-block:: default


    raster,vector = datasets.historicalMap()
    field = 'Class'







Create CV
-------------------------------------------


.. code-block:: default


    SKF = RandomStratifiedKFold(n_splits=2,
                    random_state=12,verbose=False)







Initialize Random-Forest and metrics
--------------------------------------


.. code-block:: default


    classifier = RandomForestClassifier(random_state=12,n_jobs=-1)

    # 
    kappa = metrics.make_scorer(metrics.cohen_kappa_score)
    f1_mean = metrics.make_scorer(metrics.f1_score,average='micro')
    scoring = dict(kappa=kappa,f1_mean=f1_mean,accuracy='accuracy')







Start learning
---------------------------
sklearn will compute different metrics, but will keep best results from kappa (refit='kappa')


.. code-block:: default

    LAP = learnAndPredict(n_jobs=-1,verbose=1)
    LAP.learnFromRaster(raster,vector,field,cv=SKF,
                        classifier=classifier,param_grid=dict(n_estimators=[100,200]),
                        scoring=scoring,refit='kappa')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Reading raster values...  [........................................]0%    Reading raster values...  [##################......................]45%    Reading raster values...  [####################################....]90%    Reading raster values...  [########################################]100%
    Fitting 2 folds for each of 2 candidates, totalling 4 fits
    best score : 0.9425228315087403
    best n_estimators : 200


Read the model
-------------------


.. code-block:: default

    print(LAP.model)
    print(LAP.model.cv_results_)
    print(LAP.model.best_score_)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

<<<<<<< HEAD
    GridSearchCV(cv=<museotoolbox.cross_validation.RandomStratifiedKFold object at 0x7f1c7ebff668>,
=======
    GridSearchCV(cv=<museotoolbox.cross_validation.RandomStratifiedKFold object at 0x7fc4ff3734e0>,
>>>>>>> develop
           error_score='raise',
           estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=-1,
                oob_score=False, random_state=12, verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=-1,
           param_grid={'n_estimators': [100, 200]}, pre_dispatch='2*n_jobs',
           refit='kappa', return_train_score='warn',
           scoring={'kappa': make_scorer(cohen_kappa_score), 'f1_mean': make_scorer(f1_score, average=micro), 'accuracy': 'accuracy'},
           verbose=1)
<<<<<<< HEAD
    {'mean_fit_time': array([0.71377909, 1.3084172 ]), 'std_fit_time': array([0.05055964, 0.00170743]), 'mean_score_time': array([0.60094583, 0.98617554]), 'std_score_time': array([0.04734552, 0.05384016]), 'param_n_estimators': masked_array(data=[100, 200],
=======
    {'mean_fit_time': array([0.99468303, 1.94645405]), 'std_fit_time': array([0.11697912, 0.00841641]), 'mean_score_time': array([0.78411162, 0.61318445]), 'std_score_time': array([0.0514828 , 0.00171876]), 'param_n_estimators': masked_array(data=[100, 200],
>>>>>>> develop
                 mask=[False, False],
           fill_value='?',
                dtype=object), 'params': [{'n_estimators': 100}, {'n_estimators': 200}], 'split0_test_kappa': array([0.94197678, 0.9425443 ]), 'split1_test_kappa': array([0.94276653, 0.94250136]), 'mean_test_kappa': array([0.94237166, 0.94252283]), 'std_test_kappa': array([3.94874201e-04, 2.14729960e-05]), 'rank_test_kappa': array([2, 1], dtype=int32), 'split0_train_kappa': array([0.99644289, 0.99644289]), 'split1_train_kappa': array([0.99644167, 0.9964414 ]), 'mean_train_kappa': array([0.99644228, 0.99644214]), 'std_train_kappa': array([6.08047080e-07, 7.44183416e-07]), 'split0_test_f1_mean': array([0.96662449, 0.96694084]), 'split1_test_f1_mean': array([0.96694084, 0.96678266]), 'mean_test_f1_mean': array([0.96678266, 0.96686175]), 'std_test_f1_mean': array([1.58177792e-04, 7.90888959e-05]), 'rank_test_f1_mean': array([2, 1], dtype=int32), 'split0_train_f1_mean': array([0.99794466, 0.99794466]), 'split1_train_f1_mean': array([0.99794466, 0.99794466]), 'mean_train_f1_mean': array([0.99794466, 0.99794466]), 'std_train_f1_mean': array([0., 0.]), 'split0_test_accuracy': array([0.96662449, 0.96694084]), 'split1_test_accuracy': array([0.96694084, 0.96678266]), 'mean_test_accuracy': array([0.96678266, 0.96686175]), 'std_test_accuracy': array([1.58177792e-04, 7.90888959e-05]), 'rank_test_accuracy': array([2, 1], dtype=int32), 'split0_train_accuracy': array([0.99794466, 0.99794466]), 'split1_train_accuracy': array([0.99794466, 0.99794466]), 'mean_train_accuracy': array([0.99794466, 0.99794466]), 'std_train_accuracy': array([0., 0.])}
    0.9425228315087403


Get F1 for every class from best params
-----------------------------------------------


.. code-block:: default


    for stats in LAP.getStatsFromCV(confusionMatrix=False,F1=True):
        print(stats['F1'])
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [0.97660278 0.92105263 0.99824407 0.89748549 0.        ]
    [0.97702218 0.92128028 0.99737303 0.89320388 0.        ]


Get each confusion matrix from folds
-----------------------------------------------


.. code-block:: default


    for stats in LAP.getStatsFromCV(confusionMatrix=True):
        print(stats['confusionMatrix'])
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[3694   67    1    9    0]
     [  82 1050    0   14    0]
     [   2    0 1137    0    0]
     [  12   17    1  232    0]
     [   4    0    0    0    0]]
    [[3678   79    2   12    0]
     [  69 1065    1   11    0]
     [   0    0 1139    0    0]
     [   8   21    3  230    0]
     [   3    1    0    0    0]]


Save each confusion matrix from folds
-----------------------------------------------


.. code-block:: default


    LAP.saveCMFromCV('/tmp/testMTB/',prefix='RS50_')







Predict map
---------------------------


.. code-block:: default

    
    LAP.predictRaster(raster,'/tmp/classification.tif',
                      confidence='/tmp/confidence.tif',
                      confidencePerClass='/tmp/confidencePerClass.tif')





.. rst-class:: sphx-glr-script-out
<<<<<<< HEAD

 Out:

 .. code-block:: none

    Total number of blocks : 15
    Detected 5 band(s) for function predictConfidencePerClass.
=======

 Out:

 .. code-block:: none

    Total number of blocks : 15
    Detected 1 band for function predictArray.
    Detected 5 bands for function predictConfidencePerClass.
    Detected 1 band for function predictConfidenceOfPredictedClass.
>>>>>>> develop
    Prediction... [........................................]0%    Prediction... [##......................................]6%    Prediction... [#####...................................]13%    Prediction... [########................................]20%    Prediction... [##########..............................]26%    Prediction... [#############...........................]33%    Prediction... [################........................]40%    Prediction... [##################......................]46%    Prediction... [#####################...................]53%    Prediction... [########################................]60%    Prediction... [##########################..............]66%    Prediction... [#############################...........]73%    Prediction... [################################........]80%    Prediction... [##################################......]86%    Prediction... [#####################################...]93%    Prediction... [########################################]100%
    Saved /tmp/classification.tif using function predictArray
    Saved /tmp/confidencePerClass.tif using function predictConfidencePerClass
    Saved /tmp/confidence.tif using function predictConfidenceOfPredictedClass


Plot example


.. code-block:: default


    from matplotlib import pyplot as plt
    import gdal
    src=gdal.Open('/tmp/classification.tif')
    plt.imshow(src.GetRasterBand(1).ReadAsArray(),cmap=plt.get_cmap('tab20'))
    plt.axis('off')
    plt.show()



.. image:: /auto_examples/learn_tools/images/sphx_glr_learnWithRFandRS50_001.png
    :class: sphx-glr-single-img




<<<<<<< HEAD
**Total running time of the script:** ( 0 minutes  25.753 seconds)
=======

.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  37.054 seconds)
>>>>>>> develop


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
