.. note::
    :class: sphx-glr-download-link-note

    Click :ref:`here <sphx_glr_download_auto_examples_cross_validation_LeaveOneOutPerClass.py>` to download the full example code
.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_cross_validation_LeaveOneOutPerClass.py:


Leave One Out Per Class (LOOPC)
======================================================

This example shows how to make a Leave One Out for each class.


Import librairies
-------------------------------------------


.. code-block:: default


    from museotoolbox.cross_validation import LeaveOneOutPerClass
    from museotoolbox import datasets







Load HistoricalMap dataset
-------------------------------------------


.. code-block:: default


    X,y = datasets.historicalMap(return_X_y=True)







Create CV
-------------------------------------------


.. code-block:: default

    LOOPC = LeaveOneOutPerClass(random_state=8,verbose=False)
    for tr,vl in LOOPC.split(X=None,y=y):
        print(tr,vl)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [ 1599  1600  1601 ...  9560  9561 10321] [ 6183  8386  2217  8824 10322]
    [ 1599  1600  1601 ...  9560 10321 10322] [12584  1472   832  8885  9561]
    [ 1599  1600  1601 ...  9561 10321 10322] [3488 1409 2733 8843 9508]
    [ 1599  1600  1601 ...  9561 10321 10322] [1838  721 2874 9002 9509]
    [ 1599  1600  1601 ...  9561 10321 10322] [6302 1550 2814 8964 9457]
    [ 1599  1600  1601 ...  9561 10321 10322] [3808  677 2899 9135 9560]
    [ 1599  1600  1601 ...  9561 10321 10322] [ 3487 10243  2828  1291  9458]
    [ 1599  1600  1601 ...  9560  9561 10322] [ 3988  8369   690  8979 10321]


.. note::
   Split is made to generate each fold


.. code-block:: default


    # Show label

    for tr,vl in LOOPC.split(X=None,y=y):
        print(y[vl])
    




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [1 2 3 4 5]
    [1 2 3 4 5]
    [1 2 3 4 5]
    [1 2 3 4 5]
    [1 2 3 4 5]
    [1 2 3 4 5]
    [1 2 3 4 5]
    [1 2 3 4 5]


Save each train/valid fold in a file
-------------------------------------------
In order to translate polygons into points (each points is a pixel in the raster)
we use sampleExtraction from vector_tools to generate a temporary vector.


.. code-block:: default


    trvl = LOOPC.saveVectorFiles(datasets.historicalMap()[1],'Class',outVector='/tmp/LOO.gpkg')
    for tr,vl in trvl:
        print(tr,vl)
 




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Warning : This function generates vector files according to your vector.
            The number of features may differ from the number of pixels used in classification.
            If you want to save every ROI pixels in the vector, please use vector_tools.sampleExtraction before.
    /tmp/LOO_train_0.gpkg /tmp/LOO_valid_0.gpkg
    /tmp/LOO_train_1.gpkg /tmp/LOO_valid_1.gpkg


Plot example on how a polygon was splitted


.. code-block:: default


    import ogr
    import numpy as np    
    from matplotlib import pyplot as plt
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches

    # Prepare figure
    plt.ioff()
    ax=plt.subplot(1,1,1)
    ax = plt.gca()


    xBounds,yBounds=[[],[]]

    for idx,vector in enumerate([tr,vl]):
        # Read all features in layer and store as paths    
        ds = ogr.Open(vector)
        lyr = ds.GetLayer(0)
    
        for feat in lyr:
            paths = []
            codes = []
            all_x = []
            all_y = []
        
            for geom in feat.GetGeometryRef():
                x = [geom.GetX(j) for j in range(geom.GetPointCount())]
                y = [geom.GetY(j) for j in range(geom.GetPointCount())]
                print(y)
                codes += [mpath.Path.MOVETO] + \
                                 (len(x)-1)*[mpath.Path.LINETO]
                all_x += x
                all_y += y
            path = mpath.Path(np.column_stack((all_x,all_y)), codes)
            paths.append(path)
                
            # Add paths as patches to axes
            for path in paths:
                if idx==0:
                    ax.add_patch(mpatches.PathPatch(path,color='C0'))
                else:
                    ax.add_patch(mpatches.PathPatch(path,color='C1'))
                
            xBounds.append([np.min(all_x),np.max(all_x)])
            yBounds.append([np.min(all_y),np.max(all_y)])
       

    ax.set_xlim(np.min(np.array(xBounds)[:,0]),np.max(np.array(xBounds)[:,1]))
    ax.set_ylim(np.min(np.array(yBounds)[:,0]),np.max(np.array(yBounds)[:,1]))


    legend = [mpatches.Patch(color='C0', label='Train'),mpatches.Patch(color='C1', label='Valid')]
    plt.legend(handles=legend)

    plt.show()



.. image:: /auto_examples/cross_validation/images/sphx_glr_LeaveOneOutPerClass_001.png
    :class: sphx-glr-single-img


.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [6262784.551231805, 6262838.42294402, 6262676.807807376, 6262516.988394474, 6262423.610759968, 6262466.708129739, 6262490.052538366, 6262570.860106688, 6262630.118990123, 6262689.3778735595, 6262736.066690812, 6262784.551231805]
    [6261716.09560622, 6261823.839030649, 6261866.93640042, 6261881.302190344, 6261795.107450801, 6261705.321263777, 6261626.309419196, 6261716.09560622]
    [6263201.159139596, 6263294.536774101, 6263201.159139596, 6263095.211438907, 6263123.943018755, 6263206.546310817, 6263201.159139596]
    [6261317.444935831, 6261321.036383312, 6261231.250196288, 6260978.053148881, 6260836.190973383, 6261089.388020791, 6261317.444935831]
    [6260469.863330324, 6260491.41201521, 6260444.7231979575, 6260263.35510017, 6260270.537995132, 6260371.098524598, 6260469.863330324]
    [6261202.518616446, 6261058.860717208, 6260979.848872627, 6261044.494927284, 6261202.518616446]
    [6262274.163368889, 6262241.951552138, 6261984.25701813, 6262048.680651632, 6262274.163368889]
    [6261412.497270802, 6261299.755912174, 6261042.061378167, 6261138.696828419, 6261412.497270802]
    [6260184.9222912425, 6260375.10555657, 6260329.626949644, 6260127.040427882, 6260109.124613032, 6260080.183681352, 6260184.9222912425]
    [6260061.234261801, 6260063.301471206, 6260041.595772446, 6260041.595772446, 6260061.234261801]
    [6260177.678471946, 6260055.838596495, 6259902.352000152, 6260030.521219779, 6260177.678471946]
    [6261987.03839072, 6261797.347998465, 6261707.408588345, 6261800.61852247, 6261987.03839072]
    [6262775.572613105, 6262795.32557425, 6262578.043001652, 6262534.94563188, 6262543.924250583, 6262651.667675012, 6262775.572613105]
    [6262070.720716621, 6262143.436502038, 6262178.213616802, 6261975.87403999, 6261963.227816439, 6261824.119357381, 6261754.565127851, 6261839.927136819, 6261950.581592889, 6261994.843375316, 6262070.720716621]
    [6262696.957562467, 6262663.549399839, 6262569.107093946, 6262605.727579905, 6262639.135742533, 6262723.29861377, 6262716.873967111, 6262696.957562467]
    [6259458.086246143, 6259447.400017273, 6259442.95530261, 6259454.398078657, 6259457.424267364, 6259458.086246143]
    [6262895.191637615, 6262817.727491866, 6262671.406327676, 6262779.610213798, 6262895.191637615]



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.318 seconds)


.. _sphx_glr_download_auto_examples_cross_validation_LeaveOneOutPerClass.py:


.. only :: html

 .. container:: sphx-glr-footer
    :class: sphx-glr-footer-example



  .. container:: sphx-glr-download

     :download:`Download Python source code: LeaveOneOutPerClass.py <LeaveOneOutPerClass.py>`



  .. container:: sphx-glr-download

     :download:`Download Jupyter notebook: LeaveOneOutPerClass.ipynb <LeaveOneOutPerClass.ipynb>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.readthedocs.io>`_
