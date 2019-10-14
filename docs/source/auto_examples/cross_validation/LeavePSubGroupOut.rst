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


.. code-block:: default


    from museotoolbox.cross_validation import LeavePSubGroupOut
    from museotoolbox import datasets,raster_tools
    import numpy as np







Load HistoricalMap dataset
-------------------------------------------


.. code-block:: default


    raster,vector = datasets.historicalMap(low_res=True)
    field = 'Class'
    group = 'uniquefid'







Create CV
-------------------------------------------


.. code-block:: default

    valid_size = 0.5 # Means 50%
    LPSGO = LeavePSubGroupOut(valid_size = valid_size,
                              random_state=12,verbose=False)
    






Extract X,y and group.
-------------------------------------------


.. code-block:: default


    X,y,g=raster_tools.getSamplesFromROI(raster,vector,field,group)







.. note::
   Split is made to generate each fold


.. code-block:: default


    for tr,vl in LPSGO.split(X,y,g):
        print(tr.shape,vl.shape)

    print('y label with number of samples')
    print(np.unique(y[tr],return_counts=True))




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (1498,) (1664,)
    (1791,) (1371,)
    y label with number of samples
    (array([1, 2, 3, 4, 5]), array([961, 373, 359,  97,   1]))


Differences with scikit-learn
-------------------------------------------


.. code-block:: default

    from sklearn.model_selection import LeavePGroupsOut
    # You need to specify the number of groups

    LPGO = LeavePGroupsOut(n_groups=2)
    for tr,vl in LPGO.split(X,y,g):
        print(tr.shape,vl.shape)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (2505,) (657,)
    (2662,) (500,)
    (2212,) (950,)
    (2660,) (502,)
    (2551,) (611,)
    (2726,) (436,)
    (2488,) (674,)
    (2578,) (584,)
    (2658,) (504,)
    (2695,) (467,)
    (2791,) (371,)
    (2756,) (406,)
    (2790,) (372,)
    (2560,) (602,)
    (2665,) (497,)
    (2583,) (579,)
    (2745,) (417,)
    (2295,) (867,)
    (2743,) (419,)
    (2634,) (528,)
    (2809,) (353,)
    (2571,) (591,)
    (2661,) (501,)
    (2741,) (421,)
    (2778,) (384,)
    (2874,) (288,)
    (2839,) (323,)
    (2873,) (289,)
    (2643,) (519,)
    (2748,) (414,)
    (2666,) (496,)
    (2452,) (710,)
    (2900,) (262,)
    (2791,) (371,)
    (2966,) (196,)
    (2728,) (434,)
    (2818,) (344,)
    (2898,) (264,)
    (2935,) (227,)
    (3031,) (131,)
    (2996,) (166,)
    (3030,) (132,)
    (2800,) (362,)
    (2905,) (257,)
    (2823,) (339,)
    (2450,) (712,)
    (2341,) (821,)
    (2516,) (646,)
    (2278,) (884,)
    (2368,) (794,)
    (2448,) (714,)
    (2485,) (677,)
    (2581,) (581,)
    (2546,) (616,)
    (2580,) (582,)
    (2350,) (812,)
    (2455,) (707,)
    (2373,) (789,)
    (2789,) (373,)
    (2964,) (198,)
    (2726,) (436,)
    (2816,) (346,)
    (2896,) (266,)
    (2933,) (229,)
    (3029,) (133,)
    (2994,) (168,)
    (3028,) (134,)
    (2798,) (364,)
    (2903,) (259,)
    (2821,) (341,)
    (2855,) (307,)
    (2617,) (545,)
    (2707,) (455,)
    (2787,) (375,)
    (2824,) (338,)
    (2920,) (242,)
    (2885,) (277,)
    (2919,) (243,)
    (2689,) (473,)
    (2794,) (368,)
    (2712,) (450,)
    (2792,) (370,)
    (2882,) (280,)
    (2962,) (200,)
    (2999,) (163,)
    (3095,) (67,)
    (3060,) (102,)
    (3094,) (68,)
    (2864,) (298,)
    (2969,) (193,)
    (2887,) (275,)
    (2644,) (518,)
    (2724,) (438,)
    (2761,) (401,)
    (2857,) (305,)
    (2822,) (340,)
    (2856,) (306,)
    (2626,) (536,)
    (2731,) (431,)
    (2649,) (513,)
    (2814,) (348,)
    (2851,) (311,)
    (2947,) (215,)
    (2912,) (250,)
    (2946,) (216,)
    (2716,) (446,)
    (2821,) (341,)
    (2739,) (423,)
    (2931,) (231,)
    (3027,) (135,)
    (2992,) (170,)
    (3026,) (136,)
    (2796,) (366,)
    (2901,) (261,)
    (2819,) (343,)
    (3064,) (98,)
    (3029,) (133,)
    (3063,) (99,)
    (2833,) (329,)
    (2938,) (224,)
    (2856,) (306,)
    (3125,) (37,)
    (3159,) (3,)
    (2929,) (233,)
    (3034,) (128,)
    (2952,) (210,)
    (3124,) (38,)
    (2894,) (268,)
    (2999,) (163,)
    (2917,) (245,)
    (2928,) (234,)
    (3033,) (129,)
    (2951,) (211,)
    (2803,) (359,)
    (2721,) (441,)
    (2826,) (336,)


With GroupShuffleSplit, won't keep the percentage per subgroup
This generate unbalanced classes


.. code-block:: default

    
    from sklearn.model_selection import GroupShuffleSplit
    GSS = GroupShuffleSplit(test_size=0.5,n_splits=2)
    for tr,vl in GSS.split(X,y,g):
        print(tr.shape,vl.shape)

    print('y label with number of samples')
    print(np.unique(y[tr],return_counts=True))





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    (1718,) (1444,)
    (1904,) (1258,)
    y label with number of samples
    (array([1, 2, 3, 5]), array([1237,  307,  359,    1]))


Plot example in image


.. code-block:: default

    from __drawCVmethods import plotMethod
    plotMethod('SKF-group')


.. image:: /auto_examples/cross_validation/images/sphx_glr_LeavePSubGroupOut_001.png
    :class: sphx-glr-single-img





.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.162 seconds)


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
