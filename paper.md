---
title: 'Museo ToolBox : a remote sensing python library.'

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

date: 6 December 2019

bibliography: paper.bib

---

# Summary

`Museo ToolBox` is a remote sensing python library.
Based on the fact that a majority of the needs in machine learning requires knowledge on how to transform your data and since it uses a lot of similar lines of codes on various projects but for the same usage (e.g., for reading and writing the raster block per block, computing a spectral index, fitting a model...), we offer with this library a new approach to compute very easily functions on a raster.
For example, as in our field a recurrent usage is to fit a model and predict or to use some functions like one to compute for example a spectral index, `Museo ToolBox` automatically transforms the raster to match your needs (for learning a model, the user needs an array with one line per pixel and its features as columns). Other modules help users to follow state-of-the-art methods, such as spatial cross-validation, or learning methods with grid search and latest algorithms from scikit-learn.

Museo ToolBox's goal is to make working with raster data very easier for scientists or students to promote the use of spatial cross-validation.

A [full documentation is available online on read the docs](http://museotoolbox.readthedocs.io/).

# Museo ToolBox functionnalities

`Museo ToolBox` is organized into several modules :

- geo_tools : raster and vector processing
- cross-validation : stratified cross-validation compatible with scikit-learn
- ai : machine learning module
- charts : plot confusion matrix with F1 score, mean, or producer/user's accuracy.
- stats : compute stats (like Moran's Index, confusion matrix, commision/omission) or extract truth and predicted label from confusion matrix.

Here are some main usages of `Museo ToolBox` :
1. [Read and write a raster block per block using your own function](RasterMathLink).
2. [Generate a cross-validation, including spatial cross-validation](CrossValLink).
3. [Save the cross-validation in spatial vector files (shp, gpkg, sqlite...)](SaveToVectorLink).
4. [Learn with scikit-learn algorithms and extract accuracy from each cross-validation fold](SuperLearnerLink).
5. [Plot confusion matrix and add f1 score or producer/user accuracy](ChartsLink).
6. [Get the y_true and and y_predicted labels from a confusion matrix](retrieve_y_Link).

## RasterMath

Available in `museotoolbox.geo_tools.RasterMath`, RasterMath class is the keystone of ``Museo ToolBox``.

The question is simple : How can the transposition of array-compatible functions to raster compatibility be simplified ? The idea behind ``RasterMath`` is, if your function works with an array, then it will work directly in RasterMath.

So, what does ``RasterMath`` really do ? The answer is as simple as the question : the user only work with array, so she or he doesn't have to manage the reading and writing process, the no-data management, the compression or the projection.

The objective of RasterMath is to let the user only focus on its array-compatible function, and to let RasterMath manage the raster part.

[Go to RasterMath documentation and examples](RasterMathLink)



## ai

The machine learning module is natively built to manage algorithm
from the ``scikit-learn`` using state of the art methods and good pratices (such as standardizing the input data). ``SuperLearner`` class optimizes the fit process by a grid search. There is also a Sequential Feature Selection protocol which supports number of components (e.g. a date is composed of 4 features, and you want to select the 4 features at once).

[Go to SuperLearner documentation and examples](SuperLearnerLink)

## Spatial cross-validation

``Museo ToolBox`` produces only stratified cross-validation, which means the split is made by respecting the size per class and not for the whole dataset.
For example the Leave-One-Out method will keep one sample of validation per class. As stated by [@fassnacht_2016] *"validation strategies that consider the location of the reference data
during the data-splitting are recommended"*.

- Spatial Leave-One-Out Method [@le_rest_2013; @karasiak_2019]
- Spatial Leave-Aside-Out [@engler_2013]
- Spatial Leave-One-SubGroup-Out (using centroids to select one and remove other polygons inside a specified distance buffer)

[Go to cross-validation documentation and examples](crossVal)

# Figures

A short figure presents modules with their classes and functions :

![Museo ToolBox uml.](metadata/uml.png)
