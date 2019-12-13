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


    from museotoolbox.cross_validation import LeaveOneOut
    from museotoolbox import datasets







Load HistoricalMap dataset
-------------------------------------------


.. code-block:: default


    X,y = datasets.load_historical_data(return_X_y=True,low_res=True)







Create CV
-------------------------------------------


.. code-block:: default

    LOOPC = LeaveOneOut(random_state=8,verbose=False)
    for tr,vl in LOOPC.split(X=None,y=y):
        print(tr,vl)





.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    [   0    1    2 ... 2961 3023 3160] [1864  674 3131 2910 3161]
    [   0    1    2 ... 2961 3023 3161] [ 278  453  301  508 3160]
    [   0    1    2 ... 2961 3160 3161] [2538  661 1505 2922 3023]


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


Save each train/valid fold in a file
-------------------------------------------
In order to translate polygons into points (each points is a pixel in the raster)
we use sampleExtraction from vector_tools to generate a temporary vector.


.. code-block:: default


    trvl = LOOPC.save_to_vector(datasets.load_historical_data()[1],'Class',out_vector='/tmp/LOO.gpkg')
    for tr,vl in trvl:
        print(tr,vl)

 




.. rst-class:: sphx-glr-script-out

 Out:

 .. code-block:: none

    Warning : This function generates vector files according to your vector.
            The number of features may differ from the number of pixels used in classification.
            If you want to save every ROI pixels in the vector, please use processing.sample_extraction before.
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

    [6262784.551231794, 6262838.4229440065, 6262676.8078073645, 6262516.988394462, 6262423.610759954, 6262466.708129728, 6262490.052538353, 6262570.860106677, 6262630.118990112, 6262689.377873545, 6262736.066690799, 6262784.551231794]
    [6261716.095606207, 6261823.839030637, 6261866.936400408, 6261881.302190331, 6261795.10745079, 6261705.321263766, 6261626.309419184, 6261716.095606207]
    [6263201.159139586, 6263294.53677409, 6263201.159139586, 6263095.211438895, 6263123.943018742, 6263206.546310806, 6263201.159139586]
    [6261317.444935818, 6261321.036383301, 6261231.2501962725, 6260978.0531488685, 6260836.190973371, 6261089.388020779, 6261317.444935818]
    [6260469.863330311, 6260491.412015197, 6260444.723197945, 6260263.355100158, 6260270.537995119, 6260371.098524586, 6260469.863330311]
    [6261202.518616433, 6261058.860717197, 6260979.848872616, 6261044.494927271, 6261202.518616433]
    [6262274.163368877, 6262241.951552127, 6261984.257018118, 6262048.680651621, 6262274.163368877]
    [6261412.49727079, 6261299.7559121605, 6261042.061378154, 6261138.696828406, 6261412.49727079]
    [6260184.92229123, 6260375.105556559, 6260329.626949631, 6260127.040427871, 6260109.124613019, 6260080.183681339, 6260184.92229123]
    [6260061.2342617875, 6260063.301471193, 6260041.595772434, 6260041.595772435, 6260061.2342617875]
    [6260177.678471933, 6260055.838596482, 6259902.352000138, 6260030.521219766, 6260177.678471933]
    [6261987.038390707, 6261797.347998453, 6261707.408588332, 6261800.618522459, 6261987.038390707]
    [6262775.572613094, 6262795.325574237, 6262578.043001639, 6262534.945631868, 6262543.92425057, 6262651.667675001, 6262775.572613094]
    [6262070.720716609, 6262143.436502025, 6262178.213616789, 6261975.874039977, 6261963.227816428, 6261824.119357368, 6261754.565127839, 6261839.927136808, 6261950.581592875, 6261994.843375305, 6262070.720716609]
    [6262696.957562455, 6262663.549399827, 6262569.107093934, 6262605.727579894, 6262639.13574252, 6262723.298613758, 6262716.873967098, 6262696.957562455]
    [6259458.08624613, 6259447.400017261, 6259442.955302597, 6259454.398078644, 6259457.424267353, 6259458.08624613]
    [6262895.191637604, 6262817.727491853, 6262671.406327664, 6262779.6102137845, 6262895.191637604]



.. rst-class:: sphx-glr-timing

   **Total running time of the script:** ( 0 minutes  0.226 seconds)


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
