.. role:: raw-html-m2r(raw)
   :format: html



.. image:: https://github.com/nkarasiak/MuseoToolBox/raw/master/metadata/museoToolBox_logo_128.png
   :target: https://github.com/nkarasiak/MuseoToolBox/raw/master/metadata/museoToolBox_logo_128.png
   :alt: Museo ToolBox logo



.. image:: https://api.travis-ci.org/nkarasiak/MuseoToolBox.svg?branch=master
   :target: https://travis-ci.org/nkarasiak/MuseoToolBox
   :alt: Build status


.. image:: https://readthedocs.org/projects/museotoolbox/badge/?version=latest
   :target: https://museotoolbox.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation status


.. image:: https://codecov.io/gh/nkarasiak/MuseoToolBox/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/nkarasiak/MuseoToolBox
   :alt: codecov


.. image:: https://badge.fury.io/py/museotoolbox.svg
   :target: https://badge.fury.io/py/museotoolbox
   :alt: PyPI version


.. image:: https://pepy.tech/badge/museotoolbox
   :target: https://pepy.tech/project/museotoolbox
   :alt: Downloads


.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3404729.svg
   :target: https://doi.org/10.5281/zenodo.3404728
   :alt: DOI


**Museo ToolBox** is a python library to simplify the use of raster/vector, especially for machine learning and remote sensing. It is now easy to extract raster values from vector polygons and to do some spatial/unspatial cross-validation for scikit-learn from raster.

One of the most meaningful contribution is, in my humble opinion, the `RasterMath <https://museotoolbox.readthedocs.io/en/latest/modules/processing/museotoolbox.processing.RasterMath.html>`_.

Using and citing the toolbox
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use Museo ToolBox in your research and find it useful, please cite this library using the following bibtex reference:

.. code-block:: bib

   @misc{karasiak2019mtb,
   title={Museo ToolBox : a python library for remote sensing},
   author={Karasiak Nicolas},
   url={https://github.com/nkarasiak/museotoolbox},
   year={2019},
   doi={10.5281/zenodo.3404728}
   }

What's the point ?
------------------

Today, the main usages of Museo ToolBox are :


* `museotoolbox.cross_validation <https://museotoolbox.readthedocs.io/en/latest/modules/museotoolbox.cross_validation.html#module-museotoolbox.cross_validation>`_

  * Create validation/training sets from vector, and cross-validation compatible with Scikit-Learn GridSearchCV. The aim is here to **promote the spatial cross-validation** in order to better estimate a model (with a lower spatial auto-correlation overestimation).

* `museotoolbox.processing <https://museotoolbox.readthedocs.io/en/latest/modules/museotoolbox.processing.html>`_

  * `RasterMath <https://museotoolbox.readthedocs.io/en/latest/modules/processing/museotoolbox.processing.RasterMath.html>`_\ , allows you to apply any of your array-compatible function on your raster and save it. Just load RasterMath, then it will return you the value for each pixel (in all bands) and now you can do whatever you want : predicting a model, smooth signal (whittaker, double logistic...), compute modal value... RasterMath reads and writes a raster block per block to avoid loading the full image in memory. It is compatible with every python function (including numpy) as the first and only argument RasterMath needs on your function is an array.
  * Extract bands values from vector ROI (polygons/points) (function : `extract_ROI <https://museotoolbox.readthedocs.io/en/latest/modules/processing/museotoolbox.processing.extract_ROI.html#museotoolbox.processing.extract_ROI>`_\ )  

* AI based on Scikit-Learn. `SuperLearner <https://museotoolbox.readthedocs.io/en/latest/modules/ai/museotoolbox.ai.SuperLearner.html#museotoolbox.ai.SuperLearner>`_ simplifies the use of cross-validation by extracting each accuracy (kappa,F1,OA, and above all confusion matrix) from each fold. It also eases the way to predict a raster (just give the raster name and the model).

That seems cool, but is there some help to use this ?
-----------------------------------------------------

I imagined Museo ToolBox as a tool to simplify raster processing and to promote spatial cross-validation, so of course there is some help : `a complete documentation with a lot of examples is available on readthedocs <https://museotoolbox.readthedocs.org/>`_.

I want to improve Museo ToolBox, how can I contribute ?
-------------------------------------------------------

To contribute to this package, please read the instructions in `CONTRIBUTING.rst <CONTRIBUTING.rst>`_.

How do I install it ?
---------------------

A package is available on pip :

.. code-block:: python3

   python3 -m pip install museotoolbox --user

For early-adopters, you can install the latest development version directly from git :

.. code-block:: python3

   python3 -m pip install https://github.com/nkarasiak/museotoolbox/archive/develop.zip --user

Feel free to remove the ``--user`` if you like to install the library for every user on the machine.

Who built Museo ToolBox ?
-------------------------

I am `Nicolas Karasiak <http://www.karasiak.net>`_\ , a Phd student at Dynafor Lab. I work tree species mapping from space throught dense satellite image time series, especially with Sentinel-2. A special thanks goes to `Mathieu Fauvel <http://fauvel.mathieu.free.fr/>`_ who initiated me to the beautiful world of the open-source.

Why this name ?
---------------

As Orfeo ToolBox is one my favorite and most useful library to work with raster data, I choose to name my work as Museo because in ancient Greek religion and myth, :raw-html-m2r:`<a href="https://it.wikipedia.org/wiki/Museo_(autore_mitico)">Museo is the son and disciple of Orfeo</a>`. If you want an acronym, let's say MUSEO means 'Multiple Useful Services for Earth Observation'.
