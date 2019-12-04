---
title: 'Museo ToolBox : a python remote sensing library which includes a new and definitely simple way to read and write raster.'

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

``Museo ToolBox`` is a python library directly built open the famous raster library GDAL.
In remote sensing, especially in machine learning, the majority of the work require efficience and use always the same base code (e.g., for reading and writing the raster block per block).
As in our field a recurrent usage is to fit a model and predict or to use some functions to compute for example spectral index, `Museo ToolBox` automatically transforms the raster to match your needs (for learning a model, the user needs an array with one line per pixel and its features as columns). Other modules help users to follow state-of-the-art methods, such as spatial cross-validation, or learning methods with grid search and latest algorithms from scikit-learn.

 Here are some main usages of ``Museo ToolBox`` :
 1. [Read and write a raster block per block using your own function](RasterMathLink).
 2. [Generate a cross-validation, including spatial cross-validation](CrossValLink).
 3. [Save the cross-validation in spatial vector files (shp, gpkg, sqlite...)](SaveToVectorLink).
 4. [Learn with scikit-learn algorithms and extract accuracy from each cross-validation fold](SuperLearnerLink).
 5. [Plot confusion matrix and add f1 score or producer/user accuracy](ChartsLink).
 6. [Get the y_true and and y_predicted labels from a confusion matrix](retrieve_y_Link).

# Museo ToolBox Functionnality

`Museo ToolBox` is organized into several modules :

- geo_tools : raster and vector processing
- cross-validation : stratified cross-validation compatible with scikit-learn
- ai : machine learning module
- charts : plot confusion matrix with F1 score, mean, or producer/user's accuracy.
- stats : compute stats or extract truth and predicted label from confusion matrix.

## RasterMath

Available in `museotoolbox.geo_tools.RasterMath`, RasterMath class is the keystone of ``Museo ToolBox``.

The question is simple : How can the transposition of array-compatible functions to raster compatibility be simplified ? The idea behind ``RasterMath`` is, if your function works with an array, then it will work directly in RasterMath.

So, what ``RasterMath`` really do ? The answer is as simple as the question : the user only work with array, so she or he doesn't have to manage the reading and writing procress, the no-data management, the compression or the projection.

The objective of RasterMath is to let the user only focus on its array-compatible function, and to let RasterMath manage the raster part.

## ai

The machine learning module is natively built to manage algorithm
from the ``scikit-learn`` using state of the art methods and good pratices (such as standardizing the input data). ``SuperLearner`` class optimizes the fit process by a grid search. There is also a Sequential Feature Selection protocol which supports number of components (e.g. a date is composed of 4 features, and you want to select the 4 features at once).

## Spatial cross-validation

``Museo ToolBox`` produces only stratified cross-validation, which means the split is made by class and not for the whole dataset.
For example the Leave-One-Out method will keep one sample of validation per class. As stated by [@fassnacht_2016] *"validation strategies that consider the location of the reference data
during the data-splitting are recommended"*.

- Spatial Leave-One-Out Method [@le_rest_2013,@karasiak_2019]
- Spatial Leave-Aside-Out [@engler_2013]
- Spatial Leave-One-SubGroup-Out (using centroids to select one and remove other polygons inside a specified distance buffer)

# Figures

A short uml present modules with their classes and functions :

![Museo ToolBox uml.](metadata/uml.png)
