.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_ai_learnWithRFandRS50.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_ai_learnWithRFandRS50.py:


Learn with Random-Forest and Random Sampling 50% (RS50)
========================================================

This example shows how to make a Random Sampling with 
50% for each class.


Import librairies
-------------------------------------------


.. code-block:: default


    from museotoolbox.ai import SuperLearner
    from museotoolbox.cross_validation import RandomStratifiedKFold
    from museotoolbox.processing import extract_ROI
    from museotoolbox import datasets
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics







Load HistoricalMap dataset
-------------------------------------------


.. code-block:: default


    raster,vector = datasets.load_historical_data(low_res=True)
    field = 'Class'
    X,y = extract_ROI(raster,vector,field)






Create CV
-------------------------------------------


.. code-block:: default


    SKF = RandomStratifiedKFold(n_splits=2,
                    random_state=12,verbose=False)







Initialize Random-Forest and metrics
--------------------------------------


.. code-block:: default


    classifier = RandomForestClassifier(random_state=12,n_jobs=1)

    # 
    kappa = metrics.make_scorer(metrics.cohen_kappa_score)
    f1_mean = metrics.make_scorer(metrics.f1_score,average='micro')
    scoring = dict(kappa=kappa,f1_mean=f1_mean,accuracy='accuracy')








Start learning
---------------------------
sklearn will compute different metrics, but will keep best results from kappa (refit='kappa')


.. code-block:: default

    SL = SuperLearner(classifier=classifier,param_grid = dict(n_estimators=[10]),n_jobs=1,verbose=1)

    SL.fit(X,y,cv=SKF,scoring=kappa)






.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Fitting 2 folds for each of 1 candidates, totalling 2 fits
    best score : 0.8895913422168493
    best n_estimators : 10


Read the model
-------------------


.. code-block:: default

    print(SL.model)
    print(SL.model.cv_results_)
    print(SL.model.best_score_)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    GridSearchCV(cv=<museotoolbox.cross_validation.RandomStratifiedKFold object at 0x7fab6cd34898>,
                 error_score=nan,
                 estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,
                                                  class_weight=None,
                                                  criterion='gini', max_depth=None,
                                                  max_features='auto',
                                                  max_leaf_nodes=None,
                                                  max_samples=None,
                                                  min_impurity_decrease=0.0,
                                                  min_impurity_split=None,
                                                  min_samples_leaf=1,
                                                  min_samples_split=2,
                                                  min_weight_fraction_leaf=0.0,
                                                  n_estimators=100, n_jobs=1,
                                                  oob_score=False, random_state=12,
                                                  verbose=0, warm_start=False),
                 iid='deprecated', n_jobs=1, param_grid={'n_estimators': [10]},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=make_scorer(cohen_kappa_score), verbose=1)
    {'mean_fit_time': array([0.02249706]), 'std_fit_time': array([0.00098622]), 'mean_score_time': array([0.00455117]), 'std_score_time': array([0.00014973]), 'param_n_estimators': masked_array(data=[10],
                 mask=[False],
           fill_value='?',
                dtype=object), 'params': [{'n_estimators': 10}], 'split0_test_score': array([0.90341985]), 'split1_test_score': array([0.87576284]), 'mean_test_score': array([0.88959134]), 'std_test_score': array([0.01382851]), 'rank_test_score': array([1], dtype=int32)}
    0.8895913422168493


Get F1 for every class from best params
-----------------------------------------------


.. code-block:: default


    for stats in SL.get_stats_from_cv(confusion_matrix=False,F1=True):
        print(stats['F1'])
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [0.9706191  0.85152057 0.99824253 0.73170732 0.        ]
    [0.95802728 0.81441441 0.99647887 0.703125   0.        ]


Get each confusion matrix from folds
-----------------------------------------------


.. code-block:: default


    for stats in SL.get_stats_from_cv(confusion_matrix=True):
        print(stats['confusion_matrix'])
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[925  16   0   1   0]
     [ 37 238   0  11   0]
     [  0   0 284   0   0]
     [  1  19   1  45   0]
     [  1   0   0   0   0]]
    [[913  25   0   4   0]
     [ 49 226   0  11   0]
     [  0   0 283   1   0]
     [  2  18   1  45   0]
     [  0   0   0   1   0]]


Save each confusion matrix from folds
-----------------------------------------------


.. code-block:: default


    SL.save_cm_from_cv('/tmp/testMTB/',prefix='RS50_')







Predict map
---------------------------


.. code-block:: default

    
    SL.predict_image(raster,'/tmp/classification.tif',
                      higher_confidence='/tmp/confidence.tif',
                      confidence_per_class='/tmp/confidencePerClass.tif')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 6
    Detected 1 band for function predict_array.
    Detected 5 bands for function predict_confidence_per_class.
    Detected 1 band for function predict_higher_confidence.
    Prediction... [........................................]0%    Prediction... [######..................................]16%    Prediction... [#############...........................]33%    Prediction... [####################....................]50%    Prediction... [##########################..............]66%    Prediction... [#################################.......]83%    Prediction... [########################################]100%
    Saved /tmp/classification.tif using function predict_array
    Saved /tmp/confidencePerClass.tif using function predict_confidence_per_class
    Saved /tmp/confidence.tif using function predict_higher_confidence


Plot example


.. code-block:: default


    from matplotlib import pyplot as plt
    from osgeo import gdal
    src=gdal.Open('/tmp/classification.tif')
    plt.imshow(src.GetRasterBand(1).ReadAsArray(),cmap=plt.get_cmap('tab20'))
    plt.axis('off')
    plt.show()



.. image:: /auto_examples/ai/images/sphx_glr_learnWithRFandRS50_001.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.954 seconds)


.. _sphx_glr_download_auto_examples_ai_learnWithRFandRS50.py:


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
