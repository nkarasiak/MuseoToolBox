.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_learn_tools_learnWithBandRatio.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_learn_tools_learnWithBandRatio.py:


Learn algorithm and customize your input raster without writing it on disk
=============================================================================

This example shows how to customize your raster (ndvi, smooth signal...) in the 
learning process to avoi generate a new raster.


Import librairies
-------------------------------------------


.. code-block:: default


    import numpy as np
    from museotoolbox.learn_tools import learnAndPredict
    from museotoolbox.cross_validation import RandomStratifiedKFold
    from museotoolbox import datasets
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics







Load HistoricalMap dataset
-------------------------------------------


.. code-block:: default


    raster,vector = datasets.historicalMap(low_res=True)
    field = 'Class'







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

    LAP = learnAndPredict(n_jobs=1,verbose=1)







Create or use custom function


.. code-block:: default


    def bandRatio(X,bandToKeep=[0,2]):
        # this function get the first and the last band
        X=X[:,bandToKeep].reshape(-1,len(bandToKeep))
        return X

    # add this function to learnAndPredict class
    LAP.customizeX(bandRatio)

    LAP.learnFromRaster(raster,vector,field,cv=SKF,
                        classifier=classifier,param_grid=dict(n_estimators=[10]),
                        scoring=scoring,refit='kappa')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Reading raster values...  [........................................]0%    Reading raster values...  [##......................................]5%    Reading raster values...  [####....................................]11%    Reading raster values...  [######..................................]16%    Reading raster values...  [#########...............................]22%    Reading raster values...  [###########.............................]28%    Reading raster values...  [#############...........................]33%    Reading raster values...  [###############.........................]39%    Reading raster values...  [##################......................]45%    Reading raster values...  [####################....................]50%    Reading raster values...  [######################..................]56%    Reading raster values...  [########################................]62%    Reading raster values...  [###########################.............]67%    Reading raster values...  [#############################...........]73%    Reading raster values...  [###############################.........]79%    Reading raster values...  [#################################.......]84%    Reading raster values...  [####################################....]90%    Reading raster values...  [######################################..]96%    Reading raster values...  [########################################]100%
    Fitting 2 folds for each of 1 candidates, totalling 2 fits
    best score : 0.8236149998745168
    best n_estimators : 10


Read the model
-------------------


.. code-block:: default

    print(LAP.model)
    print(LAP.model.cv_results_)
    print(LAP.model.best_score_)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    GridSearchCV(cv=<museotoolbox.cross_validation.RandomStratifiedKFold object at 0x7f39785710f0>,
           error_score='raise',
           estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=12, verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'n_estimators': [10]}, pre_dispatch='2*n_jobs',
           refit='kappa', return_train_score='warn',
           scoring={'kappa': make_scorer(cohen_kappa_score), 'f1_mean': make_scorer(f1_score, average=micro), 'accuracy': 'accuracy'},
           verbose=1)
    {'mean_fit_time': array([0.01534486]), 'std_fit_time': array([0.00073123]), 'mean_score_time': array([0.00836432]), 'std_score_time': array([0.00018895]), 'param_n_estimators': masked_array(data=[10],
                 mask=[False],
           fill_value='?',
                dtype=object), 'params': [{'n_estimators': 10}], 'split0_test_kappa': array([0.82580219]), 'split1_test_kappa': array([0.82142781]), 'mean_test_kappa': array([0.823615]), 'std_test_kappa': array([0.00218719]), 'rank_test_kappa': array([1], dtype=int32), 'split0_train_kappa': array([0.96596809]), 'split1_train_kappa': array([0.97362344]), 'mean_train_kappa': array([0.96979576]), 'std_train_kappa': array([0.00382768]), 'split0_test_f1_mean': array([0.90056998]), 'split1_test_f1_mean': array([0.89867004]), 'mean_test_f1_mean': array([0.89962001]), 'std_test_f1_mean': array([0.00094997]), 'rank_test_f1_mean': array([1], dtype=int32), 'split0_train_f1_mean': array([0.98041693]), 'split1_train_f1_mean': array([0.98483891]), 'mean_train_f1_mean': array([0.98262792]), 'std_train_f1_mean': array([0.00221099]), 'split0_test_accuracy': array([0.90056998]), 'split1_test_accuracy': array([0.89867004]), 'mean_test_accuracy': array([0.89962001]), 'std_test_accuracy': array([0.00094997]), 'rank_test_accuracy': array([1], dtype=int32), 'split0_train_accuracy': array([0.98041693]), 'split1_train_accuracy': array([0.98483891]), 'mean_train_accuracy': array([0.98262792]), 'std_train_accuracy': array([0.00221099])}
    0.8236149998745168


Get F1 for every class from best params
-----------------------------------------------


.. code-block:: default


    for stats in LAP.getStatsFromCV(confusionMatrix=False,F1=True):
        print(stats['F1'])
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [0.94179339 0.73126143 0.98954704 0.625      0.        ]
    [0.94363257 0.72202166 0.99472759 0.54700855 0.        ]


Get each confusion matrix from folds
-----------------------------------------------


.. code-block:: default


    for stats in LAP.getStatsFromCV(confusionMatrix=True):
        print(stats['confusionMatrix'])
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[898  42   0   2   0]
     [ 66 200   0  20   0]
     [  0   0 284   0   0]
     [  0  19   6  40   1]
     [  1   0   0   0   0]]
    [[904  37   0   0   1]
     [ 69 200   0  17   0]
     [  0   0 283   1   0]
     [  1  31   2  32   0]
     [  0   0   0   1   0]]


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

 Out:

 .. code-block:: none

    Total number of blocks : 18
    Detected 1 band for function predictArray.
    Detected 5 bands for function predictConfidencePerClass.
    Detected 1 band for function predictConfidenceOfPredictedClass.
    Prediction... [........................................]0%    Prediction... [##......................................]5%    Prediction... [####....................................]11%    Prediction... [######..................................]16%    Prediction... [########................................]22%    Prediction... [###########.............................]27%    Prediction... [#############...........................]33%    Prediction... [###############.........................]38%    Prediction... [#################.......................]44%    Prediction... [####################....................]50%    Prediction... [######################..................]55%    Prediction... [########################................]61%    Prediction... [##########################..............]66%    Prediction... [############################............]72%    Prediction... [###############################.........]77%    Prediction... [#################################.......]83%    Prediction... [###################################.....]88%    Prediction... [#####################################...]94%    Prediction... [########################################]100%
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



.. image:: /auto_examples/learn_tools/images/sphx_glr_learnWithBandRatio_001.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.903 seconds)


.. _sphx_glr_download_auto_examples_learn_tools_learnWithBandRatio.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: learnWithBandRatio.py <learnWithBandRatio.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: learnWithBandRatio.ipynb <learnWithBandRatio.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
