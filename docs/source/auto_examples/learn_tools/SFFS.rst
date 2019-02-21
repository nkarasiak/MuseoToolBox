.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_learn_tools_SFFS.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_learn_tools_SFFS.py:


Sequential Forward Feature Selection (SFFS)
========================================================

This example shows how to make a Random Sampling with 
50% for each class.



Import librairies
-------------------------------------------



.. code-block:: python


    from museotoolbox.learn_tools import sequentialFeatureSelection
    from museotoolbox.cross_validation import LeavePSubGroupOut
    from museotoolbox import datasets
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    import numpy as np






Load HistoricalMap dataset
-------------------------------------------



.. code-block:: python


    X,y,g = datasets.historicalMap(return_X_y_g=True)







Create CV
-------------------------------------------



.. code-block:: python


    LSGO = LeavePSubGroupOut(valid_size=0.5,n_splits=2,
                    random_state=12,verbose=False)







Initialize Random-Forest and metrics
--------------------------------------



.. code-block:: python


    classifier = RandomForestClassifier(random_state=12,n_jobs=-1)

    kappa = metrics.make_scorer(metrics.cohen_kappa_score)







Set and fit the Sequentia Feature Selection
---------------------------------------------------------------




.. code-block:: python

    SFFS = sequentialFeatureSelection(classifier,cv=LSGO,param_grid=dict(n_estimators=[100]),scoring=kappa)

    SFFS.fit(X.astype('float'),y,g,pathToSaveModels='/tmp/SFFS/')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Feature 0 already computed
    SFFS: [######..................................]16%    SFFS: [#############...........................]33%    SFFS: [####################....................]50%
    Best feature with 1 feature(s) : 2
    Best mean score : 0.715049525711
    Feature 1 already computed
    SFFS: [##########################..............]66%    SFFS: [#################################.......]83%
    Best feature with 2 feature(s) : 1
    Best mean score : 0.763682687221
    Feature 2 already computed
    SFFS: [########################################]100%

    Best feature with 3 feature(s) : 0
    Best mean score : 0.769451651927


Show best features and score



.. code-block:: python


    print('Best features are : '+str(SFFS.best_features_))
    print('Kappa are : '+str(SFFS.best_scores_))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Best features are : [2, 1, 0]
    Kappa are : [0.71504952571085778, 0.76368268722098931, 0.76945165192714637]


In order to predict every classification from the best featuree



.. code-block:: python

    SFFS.predictRasters(datasets.historicalMap()[0],'/tmp/SFFS/classification_')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ True  True  True]
    Total number of blocks : 15
    Prediction...  [........................................]0%    Prediction...  [##......................................]6%    Prediction...  [#####...................................]13%    Prediction...  [########................................]20%    Prediction...  [##########..............................]26%    Prediction...  [#############...........................]33%    Prediction...  [################........................]40%    Prediction...  [##################......................]46%    Prediction...  [#####################...................]53%    Prediction...  [########################................]60%    Prediction...  [##########################..............]66%    Prediction...  [#############################...........]73%    Prediction...  [################################........]80%    Prediction...  [##################################......]86%    Prediction...  [#####################################...]93%    Prediction...  [########################################]100%
    Saved /tmp/SFFS/classification_0.tif using function predictArray
    [ True  True False]
    Total number of blocks : 15
    Prediction...  [........................................]0%    Prediction...  [##......................................]6%    Prediction...  [#####...................................]13%    Prediction...  [########................................]20%    Prediction...  [##########..............................]26%    Prediction...  [#############...........................]33%    Prediction...  [################........................]40%    Prediction...  [##################......................]46%    Prediction...  [#####################...................]53%    Prediction...  [########################................]60%    Prediction...  [##########################..............]66%    Prediction...  [#############################...........]73%    Prediction...  [################################........]80%    Prediction...  [##################################......]86%    Prediction...  [#####################################...]93%    Prediction...  [########################################]100%
    Saved /tmp/SFFS/classification_1.tif using function predictArray
    [ True False False]
    Total number of blocks : 15
    Prediction...  [........................................]0%    Prediction...  [##......................................]6%    Prediction...  [#####...................................]13%    Prediction...  [########................................]20%    Prediction...  [##########..............................]26%    Prediction...  [#############...........................]33%    Prediction...  [################........................]40%    Prediction...  [##################......................]46%    Prediction...  [#####################...................]53%    Prediction...  [########################................]60%    Prediction...  [##########################..............]66%    Prediction...  [#############################...........]73%    Prediction...  [################################........]80%    Prediction...  [##################################......]86%    Prediction...  [#####################################...]93%    Prediction...  [########################################]100%
    Saved /tmp/SFFS/classification_2.tif using function predictArray


Plot example



.. code-block:: python


    from matplotlib import pyplot as plt
    plt.plot(np.arange(1,len(SFFS.best_scores_)+1),SFFS.best_scores_)
    plt.xlabel('Number of features')
    plt.xticks(np.arange(1,len(SFFS.best_scores_)+1))
    plt.ylabel('Kappa')
    plt.show()



.. image:: /auto_examples/learn_tools/images/sphx_glr_SFFS_001.png
    :class: sphx-glr-single-img




**Total running time of the script:** ( 0 minutes  25.231 seconds)


.. _sphx_glr_download_auto_examples_learn_tools_SFFS.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: SFFS.py <SFFS.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: SFFS.ipynb <SFFS.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
