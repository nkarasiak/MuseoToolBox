.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_charts_plotConfusion.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_charts_plotConfusion.py:


Plot confusion matrix from Cross-Validation with F1
========================================================

Plot confusion matrix from Cross-Validation, with F1 as subplot.



Import librairies
-------------------------------------------



.. code-block:: python

    from museotoolbox.learnTools import learnAndPredict
    from museotoolbox.crossValidation import RandomCV
    from museotoolbox.charts import plotConfusionMatrix
    from museotoolbox import datasets
    from sklearn.ensemble import RandomForestClassifier







Load HistoricalMap dataset
-------------------------------------------



.. code-block:: python


    raster,vector = datasets.getHistoricalMap()
    field = 'Class'






Create CV
-------------------------------------------



.. code-block:: python

    RS50 = RandomCV(valid_size=0.5,n_splits=10,
                    random_state=12,verbose=False)







Initialize Random-Forest
---------------------------



.. code-block:: python


    classifier = RandomForestClassifier()







Start learning
---------------------------



.. code-block:: python



    LAP = learnAndPredict()
    LAP.learnFromRaster(raster,vector,field,cv=RS50,
                        classifier=classifier,param_grid=dict(n_estimators=[100,200]))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Fitting 10 folds for each of 2 candidates, totalling 20 fits
    best n_estimators : 200


Get kappa from each fold
---------------------------



.. code-block:: python

  
    for kappa in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
        print(kappa)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [0.94635897652909906]
    [0.93926877916972007]
    [0.9424138426326939]
    [0.9439809301441302]
    [0.94286057027982639]
    [0.94247415327533202]
    [0.94190539222286984]
    [0.94625949356904848]
    [0.94642164578108168]
    [0.9395504758785389]


Get each confusion matrix from folds
-----------------------------------------------



.. code-block:: python

    cms = []
    for cm in LAP.getStatsFromCV(confusionMatrix=True):
        cms.append(cm)
        print(cm)
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [array([[3682,   77,    2,   10,    0],
           [  55, 1079,    1,   12,    0],
           [   2,    0, 1138,    0,    0],
           [  13,   18,    0,  232,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3687,   75,    1,    8,    0],
           [  97, 1036,    0,   13,    0],
           [   0,    0, 1139,    0,    0],
           [   5,   17,    3,  237,    0],
           [   2,    2,    0,    0,    0]])]
    [array([[3687,   70,    1,   13,    0],
           [  73, 1061,    1,   12,    0],
           [   2,    0, 1138,    0,    0],
           [   9,   29,    2,  223,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3700,   61,    2,    8,    0],
           [  84, 1047,    0,   15,    0],
           [   0,    0, 1139,    0,    0],
           [   7,   12,    2,  241,    0],
           [   3,    1,    0,    0,    0]])]
    [array([[3697,   68,    0,    6,    0],
           [  88, 1049,    0,   10,    0],
           [   0,    0, 1140,    0,    0],
           [   8,   21,    2,  232,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3700,   62,    2,    7,    0],
           [  77, 1053,    1,   15,    0],
           [   0,    0, 1139,    0,    0],
           [  16,   23,    1,  222,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3681,   75,    1,   14,    0],
           [  80, 1057,    0,   10,    0],
           [   0,    0, 1140,    0,    0],
           [  10,   17,    1,  235,    0],
           [   3,    1,    0,    0,    0]])]
    [array([[3703,   58,    2,    8,    0],
           [  59, 1063,    1,   23,    0],
           [   3,    0, 1136,    0,    0],
           [  10,   20,    3,  229,    0],
           [   4,    0,    0,    0,    0]])]
    [array([[3704,   58,    3,    6,    0],
           [  75, 1061,    1,   10,    0],
           [   0,    0, 1140,    0,    0],
           [  12,   19,    1,  231,    0],
           [   3,    0,    1,    0,    0]])]
    [array([[3686,   71,    1,   13,    0],
           [  87, 1041,    0,   18,    0],
           [   2,    0, 1137,    0,    0],
           [   9,   15,    2,  236,    0],
           [   4,    0,    0,    0,    0]])]


Plot confusion matrix
-----------------------------------------------



.. code-block:: python

    
    import numpy as np
    meanCM = np.mean(cms,axis=0)[0,:,:].astype(np.int16)
    pltCM = plotConfusionMatrix(meanCM.T) # Translate for Y = prediction and X = truth
    pltCM.addText()
    pltCM.addF1()
    pltCM.colorDiag()




.. image:: /auto_examples/charts/images/sphx_glr_plotConfusion_001.png
    :class: sphx-glr-single-img




Plot confusion matrix and normalize per class
-----------------------------------------------



.. code-block:: python


    meanCM = meanCM.astype('float') / meanCM.sum(axis=1)[:, np.newaxis]*100
    pltCM = plotConfusionMatrix(meanCM.astype(int).T)
    pltCM.addText(alpha_zero=0.3) # in order to hide a little zero values
    pltCM.addF1()
    pltCM.colorDiag()



.. image:: /auto_examples/charts/images/sphx_glr_plotConfusion_002.png
    :class: sphx-glr-single-img




**Total running time of the script:** ( 0 minutes  27.169 seconds)


.. _sphx_glr_download_auto_examples_charts_plotConfusion.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: plotConfusion.py <plotConfusion.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: plotConfusion.ipynb <plotConfusion.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
