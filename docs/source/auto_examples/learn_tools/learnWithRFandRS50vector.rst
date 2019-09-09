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


    X,y = datasets.historicalMap(return_X_y=True)







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


    LAP = learnAndPredict(n_jobs=-1)
    LAP.learnFromVector(X,y,cv=SKF,
                        classifier=classifier,param_grid=dict(n_estimators=[100,200]))







Get kappa from each fold
---------------------------


.. code-block:: default

  
    for stats in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
        print(stats['kappa'])





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    0.942544304504776
    0.9425013585127046
    0.9451833012485409
    0.9410119380054013
    0.944012651513346
    0.940879161229869
    0.9419299417406617
    0.9363722724246675
    0.9382938975182864
    0.9433560319733272


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
    [[3689   69    2   11    0]
     [  64 1067    0   15    0]
     [   0    0 1139    0    0]
     [  11   21    3  227    0]
     [   3    0    1    0    0]]
    [[3685   68    1   17    0]
     [  84 1046    1   15    0]
     [   2    0 1137    0    0]
     [   7   16    0  239    0]
     [   4    0    0    0    0]]
    [[3693   64    3   11    0]
     [  73 1057    0   16    0]
     [   1    0 1138    0    0]
     [  10   22    0  230    0]
     [   3    1    0    0    0]]
    [[3697   69    0    5    0]
     [  87 1040    1   18    0]
     [   0    0 1139    0    0]
     [   8   20    3  231    0]
     [   4    0    0    0    0]]
    [[3698   64    2    7    0]
     [  75 1060    0   11    0]
     [   3    0 1136    0    0]
     [  17   28    0  217    0]
     [   4    0    0    0    0]]
    [[3682   70    1   18    0]
     [  91 1031    1   23    0]
     [   2    0 1137    0    0]
     [   3   14    5  240    0]
     [   4    0    0    0    0]]
    [[3678   80    1   12    0]
     [  79 1049    1   17    0]
     [   1    0 1138    0    0]
     [  13   15    2  232    0]
     [   4    0    0    0    0]]
    [[3699   61    3    8    0]
     [  81 1051    0   14    0]
     [   0    0 1139    0    0]
     [  10   24    1  227    0]
     [   4    0    0    0    0]]


Only get accuracies score (OA and Kappa)
-----------------------------------------------


.. code-block:: default


    for stats in LAP.getStatsFromCV(OA=True,kappa=True,confusionMatrix=False,F1=False):
        print(stats)
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    {'kappa': 0.942544304504776, 'OA': 0.9669408415058526}
    {'kappa': 0.9425013585127046, 'OA': 0.9667826637140146}
    {'kappa': 0.9451833012485409, 'OA': 0.9683644416323948}
    {'kappa': 0.9410119380054013, 'OA': 0.9659917747548245}
    {'kappa': 0.944012651513346, 'OA': 0.9677317304650427}
    {'kappa': 0.940879161229869, 'OA': 0.9659917747548245}
    {'kappa': 0.9419299417406617, 'OA': 0.9666244859221765}
    {'kappa': 0.9363722724246675, 'OA': 0.963302752293578}
    {'kappa': 0.9382938975182864, 'OA': 0.9644099968364441}
    {'kappa': 0.9433560319733272, 'OA': 0.9674153748813666}


Save each confusion matrix from folds
-----------------------------------------------


.. code-block:: default


    LAP.saveCMFromCV('/tmp/testMTB/',prefix='SKF_',header=True)
  






Predict map
---------------------------


.. code-block:: default

    raster,_ = datasets.historicalMap()
    LAP.predictRaster(raster,'/tmp/classification.tif')





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Total number of blocks : 15
    Detected 1 band for function predictArray.
    Prediction... [........................................]0%    Prediction... [##......................................]6%    Prediction... [#####...................................]13%    Prediction... [########................................]20%    Prediction... [##########..............................]26%    Prediction... [#############...........................]33%    Prediction... [################........................]40%    Prediction... [##################......................]46%    Prediction... [#####################...................]53%    Prediction... [########################................]60%    Prediction... [##########################..............]66%    Prediction... [#############################...........]73%    Prediction... [################################........]80%    Prediction... [##################################......]86%    Prediction... [#####################################...]93%    Prediction... [########################################]100%
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

   **Total running time of the script:** ( 0 minutes  46.638 seconds)


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
