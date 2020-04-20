---
title: 'Museo ToolBox : a python library for remote sensing including a new way to handle rasters.'

tags:
  - Python
  - remote sensing
  - spatial cross-validation
  - raster
  - vector
  - autocorrelation

authors:
  - name: Nicolas Karasiak
    orcid: 0000-0002-1558-0816
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Université de Toulouse, INRAE, UMR DYNAFOR, Castanet-Tolosan, France
   index: 1

date: 13 December 2019

bibliography: paper.bib

---

# Summary

`Museo ToolBox` is a python library dedicated to the processing of georeferenced arrays, also known as rasters or images in remote sensing.

In this domain, classifying land cover type is a common and sometimes complex task, regardless of your level of expertise. Recurring procedures such as extracting Regions Of Interest (raster values from your polygon), computing spectral indices or validating a model with a cross-validation can be difficult to implement.

`Museo ToolBox` aims at simplifying the whole process by making the main treatments more accessible (extracting of Region Of Interests, fitting a model by using cross-validation, computing Normalized Difference Vegetation Index (NDVI) or various spectral indices, performing any kind of array function to the raster, etc).

The main objective of this library is to facilitate the transposition of array-like functions into an image and to promote good practices in machine learning.

To make `Museo ToolBox` easier to get started with, a [full documentation with lot of examples is available online on read the docs](http://museotoolbox.readthedocs.io/).

# Museo ToolBox in details

`Museo ToolBox` is organized into several modules (Figure 1) :

- [processing](https://museotoolbox.readthedocs.io/en/latest/modules/museotoolbox.processing.html) : raster and vector processing.
- [cross-validation](https://museotoolbox.readthedocs.io/en/latest/modules/museotoolbox.cross_validation.html) : stratified cross-validation compatible with scikit-learn.
- [ai](https://museotoolbox.readthedocs.io/en/latest/modules/museotoolbox.ai.html) :  artificial intelligence module built upon scikit-learn [@scikitlearn_2011].
- [charts](https://museotoolbox.readthedocs.io/en/latest/modules/museotoolbox.charts.html) : plot confusion matrix with F1 score or producer/user's accuracy.
- [stats](https://museotoolbox.readthedocs.io/en/latest/modules/museotoolbox.stats.html) : compute statistics (such as Moran's Index [@moran_notes_1950], confusion matrix, commision/omission) or extracting truth and predicted label from a confusion matrix.

![Museo ToolBox schema.](metadata/schema.png)



Here are some of the main usages of `Museo ToolBox` :

1. [Reading and writing a raster block per block using your own function](https://museotoolbox.readthedocs.io/en/latest/modules/processing/museotoolbox.processing.RasterMath.html).
2. [Generating cross-validation, including spatial cross-validation](https://museotoolbox.readthedocs.io/en/latest/auto_examples/index.html#cross-validation).
3. [Fitting models with scikit-learn, extracting accuracy from each cross-validation fold, and predicting raster](https://museotoolbox.readthedocs.io/en/latest/modules/ai/museotoolbox.ai.SuperLearner.html).
4. [Plotting confusion matrix and adding f1 score or producer/user accuracy](https://museotoolbox.readthedocs.io/en/latest/modules/charts/museotoolbox.charts.PlotConfusionMatrix.html#museotoolbox.charts.PlotConfusionMatrix).
5. [Getting the y_true and and y_predicted labels from a confusion matrix](https://museotoolbox.readthedocs.io/en/latest/modules/stats/museotoolbox.stats.retrieve_y_from_confusion_matrix.html).

## RasterMath

Available in `museotoolbox.processing`, `RasterMath` class is the keystone of ``Museo ToolBox``.

The question I asked myself is : How can we make it as easy as possible to implement array-like functions  to images? The idea behind ``RasterMath`` is that if the function is intended to operate with an array, it should be easy to use it with your raster using as few lines as possible.

So, what does ``RasterMath`` really do? The user only works with an array an confirms with a sample that the process is doing well, and lets `RasterMath` generalize it to the whole image. The user doesn't need to manage the raster reading and writing process, the no-data management, the compression, the number of bands, or the projection. Figure 2 explains how `RasterMath` reads a raster, performs the function, and writes it to a new raster.

The objective is to **allow the user to focus solely on the array-compatible function** while ``RasterMath`` manages the raster part.

[See RasterMath documentation and examples](https://museotoolbox.readthedocs.io/en/latest/modules/processing/museotoolbox.processing.RasterMath.html).

![RasterMath under the hood](metadata/RasterMath_schema.png)

## Artificial Intelligence

The artificial intelligence  (`ai`) module is natively built to implement ``scikit-learn`` algorithms and uses state of the art methods (such as standardizing the input data). ``SuperLearner`` class optimizes the fit process using a grid search to fix the parameters of the classifier. There is also a Sequential Feature Selection protocol which supports a number of components (e.g.  a single-date image is composed of four bands, i.e. 4 features, so a user may select 4 features at once).

[See the SuperLearner documentation and examples](https://museotoolbox.readthedocs.io/en/latest/modules/ai/museotoolbox.ai.SuperLearner.html).

## Cross-validation

``Museo ToolBox`` produces only stratified cross-validation, which means the separation between the training and the validation samples is made by respecting the size per class.
For example the Leave-One-Out method will keep one sample of validation per class. As stated by @olofsson_good_2014 *"stratified random sampling is a practical design that satisfies the
basic accuracy assessment objectives and most of the desirable design
criteria"*. For spatial cross-validation, see @karasiak_2019 inspired by @roberts_2017.

``Museo ToolBox`` offers two different kinds of cross-validation :

### Non-spatial cross-validation

- Leave-One-Out.
- Leave-One-SubGroup-Out.
- Leave-P-SubGroup-Out (Percentage of subgroup per class).
- Random Stratified K-Fold.

### Spatial cross-validation

- Spatial Leave-One-Out [@karasiak_2019].
- Spatial Leave-Aside-Out.
- Spatial Leave-One-SubGroup-Out (using centroids to select one subgroup and remove other subgroups for the same class inside a specified distance buffer).

[See the cross-validation documentation and examples](https://museotoolbox.readthedocs.io/en/latest/auto_examples/index.html#cross-validation).

# Acknowledgements

I acknowledge contributions from [Mathieu Fauvel](http://fauvel.mathieu.free.fr/), beta-testers (hey [Yousra Hamrouni](https://github.com/yousraH)), and my thesis advisors : Jean-François Dejoux, Claude Monteil and [David Sheeren](https://dsheeren.github.io/). Many thanks to Marie for proofreading.
Many thanks to Sigma students : [Hélène Ternisien de Boiville](https://github.com/HTDBD), [Arthur Duflos](https://github.com/ArthurDfs), [Sam Antonetti](https://github.com/santonetti) and [Anne-Sophie Tronc](https://github.com/AnneSophieTronc) for their implication in RasterMath improvements in early 2020.

# References
