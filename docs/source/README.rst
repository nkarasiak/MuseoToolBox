

.. image:: https://readthedocs.org/projects/museotoolbox/badge/?version=latest
   :target: https://museotoolbox.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


.. image:: https://badge.fury.io/py/museotoolbox.svg
   :target: https://badge.fury.io/py/museotoolbox
   :alt: PyPI version


.. image:: https://pepy.tech/badge/museotoolbox
   :target: https://pepy.tech/project/museotoolbox
   :alt: Downloads


.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3404729.svg
   :target: https://doi.org/10.5281/zenodo.3404728
   :alt: DOI



.. image:: https://github.com/lennepkade/MuseoToolBox/raw/master/metadata/museoToolBox_logo_128.png
   :target: https://github.com/lennepkade/MuseoToolBox/raw/master/metadata/museoToolBox_logo_128.png
   :alt: MuseoToolBox logo


**Museo ToolBox** is a python library to simplify the use of raster/vector, especially for machine learning and for remote sensing. It is now really easy to extract raster values from vector and to do some spatial/unspatial cross-validation for scikit-learn.

The other meaningful contribution is the **rasterMath** function which allow you to do whatever you like on a raster in a just few lines : compute the mean in all bands, the modal value, smooth a signal, compute the ndvi... all you have to do is to give your own function to MuseoToolBox, and **rasterMath** manage everything : the nodata value, reading the raster block per block, saving the result to a new raster with a fast compression. `Examples and code using rasterMath are available on readthedocs <https://museotoolbox.readthedocs.io/en/latest/modules/raster_tools/museotoolbox.raster_tools.rasterMath.html#museotoolbox.raster_tools.rasterMath>`_

Using and citing the toolbox
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use this toolbox in your research and find it useful, please cite MuseoToolBox using the following bibtex reference:

.. code-block::

   @misc{karasiak2019mtb,
   title={MuseoToolBox, a remote sensing python library},
   author={Karasiak Nicolas},
   url={https://github.com/lennepkade/museotoolbox},
   year={2019},
   doi={10.5281/zenodo.3404728}
   }

What's the point ?
------------------

Today, the main usages of Museo ToolBox are :


* museotoolbox.\ **cross_validation**

  * Create validation/training sets from vector, and a cross-validation directly compatible with Scikit-Learn GridSearchCV. The aim is here to **promote the spatial validation/training** in order to better estimate a model (a lower spatial auto-correlation overestimation).

* museotoolbox.\ **raster_tools**

  * Extract band value from vector ROI (polygons/points) (function : *getSamplesFromROI*\ )
  * **rasterMath**\ , allows you to do some math on your raster and save it. Just load it, rasterMath will return you the value for each pixel (in all bands) and do whatever you want : predicting a model, signal treatment (whittaker, double logistic...), modal value, mean...
    Compatible with every python function (including numpy) as the first argument an array.

* museotoolbox.\ **learn_tools**

  * Based on Scikit-Learn. Simplify the use the cross-Validations by extracting each accuracy (kappa,F1,OA, and above all confusion matrix) from each fold. Ease the way to predict a raster (just give the raster name and the model).

That seems cool, but is there some help to use this ?
-----------------------------------------------------

I imagined Museo ToolBox as a tool to promote the use of spatial cross-validation and to learn/predict from raster, so of course there is some help : `a complete documentation with a lot of examples is available on readthedocs <https://museotoolbox.readthedocs.org/>`_.

How do I install it ?
---------------------

A package is available on pip : 
``python3 -m pip install museotoolbox --user`` 

Alternatively, you can install **museotoolbox** directly from the git :
``python3 -m pip install git+https://github.com/lennepkade/MuseoToolBox.git --user``

Feel free to remove the ``--user`` if you like to install the library for every user on the machine.

Who build Museo ToolBox ?
-------------------------

I am `Nicolas Karasiak <http://www.karasiak.net>`_\ , a Phd student at Dynafor Lab. I work on the identification of tree species throught dense satellite image time series, especially with Sentinel-2. A special thanks goes to `Mathieu Fauvel <http://fauvel.mathieu.free.fr/>`_ who initiates me to the nice and open-source coding.

Why this name ?
---------------

As Orfeo ToolBox is one my favorite and most useful library to work with raster data, I choose to name my work as Museo because in ancient Greek religion and myth, `Museo is the son and disciple of Orfeo <https://it.wikipedia.org/wiki/Museo_(autore_mitico>`_\ ). If you want an acronym, let's say MUSEO means 'Multiple Useful Services for Earth Observation'.
