.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_cross_validation_LeavePSubGroupOut.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_cross_validation_LeavePSubGroupOut.py:


Leave-P-SubGroup-Out (LPSGO)
======================================================

This example shows how to make a Leave-Percent-SubGroup-Out.



Import librairies
-------------------------------------------



.. code-block:: python


    from museotoolbox.cross_validation import LeavePSubGroupOut
    from museotoolbox import datasets,raster_tools
    import numpy as np







Load HistoricalMap dataset
-------------------------------------------



.. code-block:: python


    raster,vector = datasets.historicalMap()
    field = 'Class'
    group = 'uniquefid'







Create CV
-------------------------------------------



.. code-block:: python

    valid_size = 0.5 # Means 50%
    LPSGO = LeavePSubGroupOut(valid_size = valid_size,n_splits=10,
                              random_state=12,verbose=False)
    






Extract X,y and group.
-------------------------------------------



.. code-block:: python


    X,y,g=raster_tools.getSamplesFromROI(raster,vector,field,group)







.. note::
   Split is made to generate each fold



.. code-block:: python


    for tr,vl in LPSGO.split(X,y,g):
        print(tr.shape,vl.shape)

    print('y label with number of samples')
    print(np.unique(y[tr],return_counts=True))




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (5998,) (6649,)
    (7147,) (5500,)
    (8008,) (4639,)
    (5575,) (7072,)
    (5926,) (6721,)
    (7566,) (5081,)
    (6857,) (5790,)
    (6635,) (6012,)
    (6089,) (6558,)
    (7056,) (5591,)
    y label with number of samples
    (array([1, 2, 3, 4, 5]), array([3830, 1492, 1343,  385,    6]))


Differences with scikit-learn
-------------------------------------------



.. code-block:: python

    from sklearn.model_selection import LeavePGroupsOut
    # You need to specify the number of groups
    LPGO = LeavePGroupsOut(n_groups=2)
    for tr,vl in LPGO.split(X,y,g):
        print(tr.shape,vl.shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (10030,) (2617,)
    (10644,) (2003,)
    (8842,) (3805,)
    (10635,) (2012,)
    (10205,) (2442,)
    (10901,) (1746,)
    (9953,) (2694,)
    (10300,) (2347,)
    (10630,) (2017,)
    (10781,) (1866,)
    (11160,) (1487,)
    (11026,) (1621,)
    (11164,) (1483,)
    (10230,) (2417,)
    (10668,) (1979,)
    (10321,) (2326,)
    (10989,) (1658,)
    (9187,) (3460,)
    (10980,) (1667,)
    (10550,) (2097,)
    (11246,) (1401,)
    (10298,) (2349,)
    (10645,) (2002,)
    (10975,) (1672,)
    (11126,) (1521,)
    (11505,) (1142,)
    (11371,) (1276,)
    (11509,) (1138,)
    (10575,) (2072,)
    (11013,) (1634,)
    (10666,) (1981,)
    (9801,) (2846,)
    (11594,) (1053,)
    (11164,) (1483,)
    (11860,) (787,)
    (10912,) (1735,)
    (11259,) (1388,)
    (11589,) (1058,)
    (11740,) (907,)
    (12119,) (528,)
    (11985,) (662,)
    (12123,) (524,)
    (11189,) (1458,)
    (11627,) (1020,)
    (11280,) (1367,)
    (9792,) (2855,)
    (9362,) (3285,)
    (10058,) (2589,)
    (9110,) (3537,)
    (9457,) (3190,)
    (9787,) (2860,)
    (9938,) (2709,)
    (10317,) (2330,)
    (10183,) (2464,)
    (10321,) (2326,)
    (9387,) (3260,)
    (9825,) (2822,)
    (9478,) (3169,)
    (11155,) (1492,)
    (11851,) (796,)
    (10903,) (1744,)
    (11250,) (1397,)
    (11580,) (1067,)
    (11731,) (916,)
    (12110,) (537,)
    (11976,) (671,)
    (12114,) (533,)
    (11180,) (1467,)
    (11618,) (1029,)
    (11271,) (1376,)
    (11421,) (1226,)
    (10473,) (2174,)
    (10820,) (1827,)
    (11150,) (1497,)
    (11301,) (1346,)
    (11680,) (967,)
    (11546,) (1101,)
    (11684,) (963,)
    (10750,) (1897,)
    (11188,) (1459,)
    (10841,) (1806,)
    (11169,) (1478,)
    (11516,) (1131,)
    (11846,) (801,)
    (11997,) (650,)
    (12376,) (271,)
    (12242,) (405,)
    (12380,) (267,)
    (11446,) (1201,)
    (11884,) (763,)
    (11537,) (1110,)
    (10568,) (2079,)
    (10898,) (1749,)
    (11049,) (1598,)
    (11428,) (1219,)
    (11294,) (1353,)
    (11432,) (1215,)
    (10498,) (2149,)
    (10936,) (1711,)
    (10589,) (2058,)
    (11245,) (1402,)
    (11396,) (1251,)
    (11775,) (872,)
    (11641,) (1006,)
    (11779,) (868,)
    (10845,) (1802,)
    (11283,) (1364,)
    (10936,) (1711,)
    (11726,) (921,)
    (12105,) (542,)
    (11971,) (676,)
    (12109,) (538,)
    (11175,) (1472,)
    (11613,) (1034,)
    (11266,) (1381,)
    (12256,) (391,)
    (12122,) (525,)
    (12260,) (387,)
    (11326,) (1321,)
    (11764,) (883,)
    (11417,) (1230,)
    (12501,) (146,)
    (12639,) (8,)
    (11705,) (942,)
    (12143,) (504,)
    (11796,) (851,)
    (12505,) (142,)
    (11571,) (1076,)
    (12009,) (638,)
    (11662,) (985,)
    (11709,) (938,)
    (12147,) (500,)
    (11800,) (847,)
    (11213,) (1434,)
    (10866,) (1781,)
    (11304,) (1343,)


With GroupShuffleSplit, won't keep the percentage per subgroup
This generate unbalanced classes



.. code-block:: python

    
    from sklearn.model_selection import GroupShuffleSplit
    GSS = GroupShuffleSplit(test_size=0.5,n_splits=5)
    for tr,vl in GSS.split(X,y,g):
        print(tr.shape,vl.shape)

    print('y label with number of samples')
    print(np.unique(y[tr],return_counts=True))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (7312,) (5335,)
    (4142,) (8505,)
    (5118,) (7529,)
    (4705,) (7942,)
    (3660,) (8987,)
    y label with number of samples
    (array([1, 2, 3, 4, 5]), array([1481,  801,  845,  525,    8]))


Plot example in image



.. code-block:: python

    
    import numpy as np
    from matplotlib import pyplot as plt
    plt.scatter(np.random.randint(10,20,40),np.random.randint(10,30,40),s=100,color='#1f77b4')
    plt.scatter(np.random.randint(0,10,40),np.random.randint(10,30,40),s=100,color='#1f77b4')
    plt.scatter(np.random.randint(0,10,20),np.random.randint(0,10,20),s=100,color='#ff7f0e')
    plt.scatter(np.random.randint(20,30,20),np.random.randint(10,30,20),s=100,color='#ff7f0e')
    plt.axis('off')
    plt.show()


.. image:: /auto_examples/cross_validation/images/sphx_glr_LeavePSubGroupOut_001.png
    :class: sphx-glr-single-img




**Total running time of the script:** ( 0 minutes  0.188 seconds)


.. _sphx_glr_download_auto_examples_cross_validation_LeavePSubGroupOut.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: LeavePSubGroupOut.py <LeavePSubGroupOut.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: LeavePSubGroupOut.ipynb <LeavePSubGroupOut.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
