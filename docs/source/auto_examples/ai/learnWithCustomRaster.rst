.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_ai_learnWithCustomRaster.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_ai_learnWithCustomRaster.py:


Learn algorithm and customize your input raster without writing it on disk
=============================================================================

This example shows how to customize your raster (ndvi, smooth signal...) in the 
learning process to avoi generate a new raster.


Import librairies
-------------------------------------------


.. code-block:: default


    from museotoolbox.ai import SuperLearner
    from museotoolbox.processing import extract_ROI
    from museotoolbox import datasets
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics







Load HistoricalMap dataset
-------------------------------------------


.. code-block:: default


    raster,vector = datasets.load_historical_data(low_res=True)
    field = 'Class'







Initialize Random-Forest and metrics
--------------------------------------


.. code-block:: default


    classifier = RandomForestClassifier(random_state=12,n_jobs=1)

    kappa = metrics.make_scorer(metrics.cohen_kappa_score)
    f1_mean = metrics.make_scorer(metrics.f1_score,average='micro')
    scoring = dict(kappa=kappa,f1_mean=f1_mean,accuracy='accuracy')








Start learning
---------------------------
sklearn will compute different metrics, but will keep best results from kappa (refit='kappa')


.. code-block:: default

    SL = SuperLearner(classifier=classifier,param_grid=dict(n_estimators=[10]),n_jobs=1,verbose=1)







Create or use custom function


.. code-block:: default


    def reduceBands(X,bandToKeep=[0,2]):
        # this function get the first and the last band
        X=X[:,bandToKeep].reshape(-1,len(bandToKeep))
        return X

    # add this function to learnAndPredict class
    SL.customize_array(reduceBands)

    # if you learn from vector, refit according to the f1_mean
    X,y = extract_ROI(raster,vector,field)
    SL.fit(X,y,cv=2,scoring=scoring,refit='f1_mean')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Fitting 2 folds for each of 1 candidates, totalling 2 fits
    best score : 0.90880303989867
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

    GridSearchCV(cv=<museotoolbox.cross_validation.RandomStratifiedKFold object at 0x7fab6ce14c18>,
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
                                                  min_...
                                                  min_weight_fraction_leaf=0.0,
                                                  n_estimators=100, n_jobs=1,
                                                  oob_score=False, random_state=12,
                                                  verbose=0, warm_start=False),
                 iid='deprecated', n_jobs=1, param_grid={'n_estimators': [10]},
                 pre_dispatch='2*n_jobs', refit='f1_mean', return_train_score=False,
                 scoring={'accuracy': 'accuracy',
                          'f1_mean': make_scorer(f1_score, average=micro),
                          'kappa': make_scorer(cohen_kappa_score)},
                 verbose=1)
    {'mean_fit_time': array([0.02084494]), 'std_fit_time': array([0.00128293]), 'mean_score_time': array([0.00668275]), 'std_score_time': array([0.00099218]), 'param_n_estimators': masked_array(data=[10],
                 mask=[False],
           fill_value='?',
                dtype=object), 'params': [{'n_estimators': 10}], 'split0_test_kappa': array([0.84332782]), 'split1_test_kappa': array([0.83718761]), 'mean_test_kappa': array([0.84025771]), 'std_test_kappa': array([0.00307011]), 'rank_test_kappa': array([1], dtype=int32), 'split0_test_f1_mean': array([0.91070298]), 'split1_test_f1_mean': array([0.9069031]), 'mean_test_f1_mean': array([0.90880304]), 'std_test_f1_mean': array([0.00189994]), 'rank_test_f1_mean': array([1], dtype=int32), 'split0_test_accuracy': array([0.91070298]), 'split1_test_accuracy': array([0.9069031]), 'mean_test_accuracy': array([0.90880304]), 'std_test_accuracy': array([0.00189994]), 'rank_test_accuracy': array([1], dtype=int32)}
    0.90880303989867


Get F1 for every class from best params
-----------------------------------------------


.. code-block:: default


    for stats in SL.get_stats_from_cv(confusion_matrix=False,F1=True):
        print(stats['F1'])
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [0.94246862 0.75138122 0.99647887 0.76335878 0.        ]
    [0.94099052 0.7539267  0.99300699 0.68421053 0.        ]


Get each confusion matrix from folds
-----------------------------------------------


.. code-block:: default


    for stats in SL.get_stats_from_cv(confusion_matrix=True):
        print(stats['confusion_matrix'])
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[901  40   0   0   1]
     [ 66 204   0  14   2]
     [  0   0 283   1   0]
     [  2  13   1  50   0]
     [  1   0   0   0   0]]
    [[893  47   0   2   0]
     [ 63 216   0   7   0]
     [  0   0 284   0   0]
     [  0  23   4  39   0]
     [  0   1   0   0   0]]


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



.. image:: /auto_examples/ai/images/sphx_glr_learnWithCustomRaster_001.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.967 seconds)


.. _sphx_glr_download_auto_examples_ai_learnWithCustomRaster.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: learnWithCustomRaster.py <learnWithCustomRaster.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: learnWithCustomRaster.ipynb <learnWithCustomRaster.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
