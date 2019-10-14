[![Documentation status](https://readthedocs.org/projects/museotoolbox/badge/?version=latest)](https://museotoolbox.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/museotoolbox.svg)](https://badge.fury.io/py/museotoolbox)
[![Build status](https://api.travis-ci.org/nkarasiak/MuseoToolBox.svg?branch=master)](https://travis-ci.org/nkarasiak/MuseoToolBox)
[![Downloads](https://pepy.tech/badge/museotoolbox)](https://pepy.tech/project/museotoolbox)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3404729.svg)](https://doi.org/10.5281/zenodo.3404728)

![MuseoToolBox logo](https://github.com/nkarasiak/MuseoToolBox/raw/master/metadata/museoToolBox_logo_128.png)

**MuseoToolBox** is a python library to simplify the use of raster/vector, especially for machine learning and remote sensing. It is now easy to extract raster values from vector polygons and to do some spatial/unspatial cross-validation for scikit-learn from raster.

One of the most meaningful contribution is, in my humble opinion, the [**rasterMath**](https://museotoolbox.readthedocs.io/en/latest/modules/raster_tools/museotoolbox.raster_tools.rasterMath.html#museotoolbox.raster_tools.rasterMath) class which allows you to do any expression/function on a raster in a just few lines : compute the mean in all bands, the modal value, smooth a signal, compute the ndvi... all you have to do is to give your own function to MuseoToolBox, and **rasterMath** manages everything : the nodata value, reading the raster block per block, saving the result to a new raster with a fast compression. [Examples to use rasterMath are available on readthedocs](https://museotoolbox.readthedocs.io/en/latest/modules/raster_tools/museotoolbox.raster_tools.rasterMath.html#museotoolbox.raster_tools.rasterMath).

### Using and citing the toolbox

If you use MuseoToolBox in your research and find it useful, please cite this library using the following bibtex reference:
```
@misc{karasiak2019mtb,
title={MuseoToolBox, a remote sensing python library},
author={Karasiak Nicolas},
url={https://github.com/nkarasiak/museotoolbox},
year={2019},
doi={10.5281/zenodo.3404728}
}
```

## What's the point ?

Today, the main usages of MuseoToolBox are :
-  [museotoolbox.**cross_validation** ](https://museotoolbox.readthedocs.io/en/latest/modules/museotoolbox.cross_validation.html)
  - Create validation/training sets from vector, and cross-validation compatible with Scikit-Learn GridSearchCV. The aim is here to **promote the spatial cross-validation** in order to better estimate a model (with a lower spatial auto-correlation overestimation).
- [museotoolbox.**raster_tools**](https://museotoolbox.readthedocs.io/en/latest/modules/museotoolbox.raster_tools.html)
  - Extract bands values from vector ROI (polygons/points) (function : [*getSamplesFromROI*](https://museotoolbox.readthedocs.io/en/latest/modules/raster_tools/museotoolbox.raster_tools.getSamplesFromROI.html))
  - [**rasterMath**](https://museotoolbox.readthedocs.io/en/latest/modules/raster_tools/museotoolbox.raster_tools.rasterMath.html#museotoolbox.raster_tools.rasterMath), allows you to do some math on your raster and save it. Just load rasterMath, then it will return you the value for each pixel (in all bands) and now you can do whatever you want : predicting a model, smooth signal (whittaker, double logistic...), modal value, mean... rasterMath read and write block per block to avoid loading the full image in memory. It is compatible with every python function (including numpy) as the first and only argument needed is an array.
- [museotoolbox.**learn_tools**](https://museotoolbox.readthedocs.io/en/latest/modules/museotoolbox.learn_tools.html)
  - Based on Scikit-Learn. [**learndAndPredict**](https://museotoolbox.readthedocs.io/en/latest/modules/learn_tools/museotoolbox.learn_tools.learnAndPredict.html) simplifies the use of cross-validation by extracting each accuracy (kappa,F1,OA, and above all confusion matrix) from each fold. It also eases the way to predict a raster (just give the raster name and the model).

## That seems cool, but is there some help to use this ?
I imagined MuseoToolBox as a tool to simplify raster processing and to promote spatial cross-validation, so of course there is some help : [a complete documentation with a lot of examples is available on readthedocs](https://museotoolbox.readthedocs.org/).

## How do I install it ?
A package is available on pip :
`python3 -m pip install museotoolbox --user`

Alternatively, you can install **museotoolbox** directly from the git :
`python3 -m pip install git+https://github.com/nkarasiak/MuseoToolBox.git --user`

Feel free to remove the `--user` if you like to install the library for every user on the machine.

## Who built MuseoToolBox ?

I am [Nicolas Karasiak](http://www.karasiak.net), a Phd student at Dynafor Lab. I work tree species mapping from space throught dense satellite image time series, especially with Sentinel-2. A special thanks goes to [Mathieu Fauvel](http://fauvel.mathieu.free.fr/) who initiated me to the beautiful world of the open-source.

## Why this name ?
As Orfeo ToolBox is one my favorite and most useful library to work with raster data, I choose to name my work as Museo because in ancient Greek religion and myth, [Museo is the son and disciple of Orfeo](https://it.wikipedia.org/wiki/Museo_(autore_mitico)). If you want an acronym, let's say MUSEO means 'Multiple Useful Services for Earth Observation'.
