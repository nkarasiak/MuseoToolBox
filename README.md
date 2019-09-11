[![Documentation Status](https://readthedocs.org/projects/museotoolbox/badge/?version=latest)](https://museotoolbox.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/museotoolbox.svg)](https://badge.fury.io/py/museotoolbox) [![Downloads](https://pepy.tech/badge/museotoolbox)](https://pepy.tech/project/museotoolbox)

![MuseoToolBox logo](https://github.com/lennepkade/MuseoToolBox/raw/master/metadata/museoToolBox_logo_128.png)

**Museo ToolBox** is a python library to simplify the use of raster/vector. One of the most important contributions is the interface between a spatial or subgroup cross-validation strategy and learning and prediction steps with Scikit-Learn. 

The other meaningful contribution is the **rasterMath** function which allow you to do whatever you like on a raster in a few lines : mean/modal/prediction/whittaker (you use your own function), and **rasterMath** manage everything : the nodata value, reading the raster block per block, saving to a new raster...

#### Using and citing the toolbox

If you use this toolbox in your research and find it useful, please cite MuseoToolBox using the following bibtex reference:
```
@misc{karasiak2019mtb,
title={MuseoToolBox, a remote sensing python library},
author={Karasiak Nicolas},
url={https://github.com/lennepkade/museotoolbox},
year={2019},
doi={10.5281/zenodo.3404728}
}
```

## What's the point ?

Today, the main usages of Museo ToolBox are :
- museotoolbox.**cross_validation**
  - Create validation/training sets from vector, and a cross-validation directly compatible with Scikit-Learn GridSearchCV. The aim is here to **promote the spatial validation/training** in order to better estimate a model (a lower spatial auto-correlation overestimation).
- museotoolbox.**raster_tools**
  - Extract band value from vector ROI (polygons/points) (function : *getSamplesFromROI*)
  - **rasterMath**, allows you to do some math on your raster and save it. Just load it, rasterMath will return you the value for each pixel (in all bands) and do whatever you want : predicting a model, signal treatment (whittaker, double logistic...), modal value, mean...
  Compatible with every python function (including numpy) as the first argument an array.
- museotoolbox.**learn_tools**
  - Based on Scikit-Learn. Simplify the use the cross-Validations by extracting each accuracy (kappa,F1,OA, and above all confusion matrix) from each fold. Ease the way to predict a raster (just give the raster name and the model).
  Also simplify the prediction of a model to a raster.

## That seems cool, but is there some help to use this ?
I imagined Museo ToolBox as a tool to promote the use of spatial cross-validation (or validation/training at least by subgroup) and to learn and predict from raster, so of course I help you : [a complete documentation with a lot of examples is available on readthedocs](https://museotoolbox.readthedocs.org/).

## How do I install it ?
A package will be available on pip : 
`python3 -m pip install museotoolbox --user` 

Alternatively, you can install **museotoolbox** directly from the git :
`python3 -m pip install git+https://github.com/lennepkade/MuseoToolBox.git --user`

Feel free to remove the `--user` if you like to install the library for every user on the machine.

## Who build Museo ToolBox ?

I am [Nicolas Karasiak](http://www.karasiak.net), a Phd student at Dynafor Lab. I work on the identification of tree species throught dense satellite image time series, especially with Sentinel-2. A special thanks goes to [Mathieu Fauvel](http://fauvel.mathieu.free.fr/) who initiates me to the nice and open-source coding.

## Why this name ?
As Orfeo ToolBox is one my favorite and most useful library to work with raster data, I choose to name my work as Museo because in ancient Greek religion and myth, [Museo is the son and disciple of Orfeo](https://it.wikipedia.org/wiki/Museo_(autore_mitico)). If you want an acronym, let's say MUSEO means 'Multiple Useful Services for Earth Observation'.
