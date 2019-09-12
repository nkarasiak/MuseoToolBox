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

    0.9339728776114595
    0.9390240071759156
    0.9342722183695072
    0.9335604525610437
    0.9349508141372171
    0.9342768194480748
    0.9336502026509483
    0.926137103513494
    0.9317242405823273
    0.9366696896132733


Get each confusion matrix from folds
-----------------------------------------------


.. code-block:: default


    for stats in LAP.getStatsFromCV(confusionMatrix=True):
        print(stats['confusionMatrix'])
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [[3699   63    2    7    0]
     [ 111 1020    0   15    0]
     [   5    0 1134    0    0]
     [  17   14    1  230    0]
     [   4    0    0    0    0]]
    [[3690   67    2   12    0]
     [  84 1044    1   17    0]
     [   1    0 1138    0    0]
     [   9   22    3  228    0]
     [   3    1    0    0    0]]
    [[3687   68    2   14    0]
     [  96 1032    0   18    0]
     [   0    0 1139    0    0]
     [  11   23    3  225    0]
     [   4    0    0    0    0]]
    [[3692   63    0   16    0]
     [ 112 1018    1   15    0]
     [   2    0 1137    0    0]
     [  12   15    1  234    0]
     [   4    0    0    0    0]]
    [[3696   60    3   12    0]
     [  99 1034    0   13    0]
     [   1    0 1138    0    0]
     [  14   27    3  218    0]
     [   3    1    0    0    0]]
    [[3704   63    0    4    0]
     [ 112 1014    1   19    0]
     [   1    0 1138    0    0]
     [  11   22    1  228    0]
     [   4    0    0    0    0]]
    [[3702   61    1    7    0]
     [ 104 1035    0    7    0]
     [   5    0 1134    0    0]
     [  18   33    0  211    0]
     [   4    0    0    0    0]]
    [[3685   68    1   17    0]
     [ 125  997    1   23    0]
     [   2    0 1137    0    0]
     [   7   17    3  235    0]
     [   4    0    0    0    0]]
    [[3684   73    1   13    0]
     [ 105 1026    1   14    0]
     [   1    0 1138    0    0]
     [  12   20    4  226    0]
     [   4    0    0    0    0]]
    [[3694   62    5   10    0]
     [  92 1043    0   11    0]
     [   0    0 1139    0    0]
     [  12   34    0  216    0]
     [   4    0    0    0    0]]


Only get accuracies score (OA and Kappa)
-----------------------------------------------


.. code-block:: default


    for stats in LAP.getStatsFromCV(OA=True,kappa=True,confusionMatrix=False,F1=False):
        print(stats)
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    {'kappa': 0.9339728776114595, 'OA': 0.9621955077507118}
    {'kappa': 0.9390240071759156, 'OA': 0.9648845302119582}
    {'kappa': 0.9342722183695072, 'OA': 0.9621955077507118}
    {'kappa': 0.9335604525610437, 'OA': 0.9618791521670358}
    {'kappa': 0.9349508141372171, 'OA': 0.9626700411262259}
    {'kappa': 0.9342768194480748, 'OA': 0.9623536855425499}
    {'kappa': 0.9336502026509483, 'OA': 0.9620373299588738}
    {'kappa': 0.926137103513494, 'OA': 0.957608351787409}
    {'kappa': 0.9317242405823273, 'OA': 0.9607719076241695}
    {'kappa': 0.9366696896132733, 'OA': 0.963619107877254}


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

   **Total running time of the script:** ( 0 minutes  2.779 seconds)


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
