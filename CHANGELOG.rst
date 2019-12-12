# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### added
- add zonal_stats example
- update docstring in ai
### changed
- modify SequentialFeatureSelection to have same fit method as SuperLearner
### removed
- changelog.txt to CHANGELOG.md

## [2.0] - 2019-12-11
### changed 
Update RasterMath and readme file.

## [2.0b1] - 2019-12-08
### changed
- Some folders have changed name :
	- raster_tools and vector_tools to processing
	- learn_tools to ai
- some functions have changed name : 
	- getSamplesFromROI to extract_values
	- historicalMap to load_historical_data
	- getDistanceMatrix to get_distance_matrix
- classes now always begin with a capital case : 
    - learnAndPredict to SuperLearner
    - rasterMath to RasterMath
    - sequentialFeatureSelection to SequentialFeatureSelection 
- Fix bug #7 : getSamplesFromROI (nowd extract_ROI) now extracts ROI values using by default memory. If it fails, it will create a temporary raster on disk then delete it when finished. 
### Removed
- Remove command lines (cli)

## [1.6.6] - 2019-11-11	
### changed
- getSamplesFromROI return list of available fields if wrong field given.
- rasterMath convert np.nan value to nodata value (if numpy >= 1.17)

## [1.6.5] - 2019-11-01
### changed
- Minor fix when using learnAndPredict with an outside customized function
- Better management fo cross-validation in learnAndPredict
- Fix minor bug using False or None value with cv in learnAndPredict
### aded
- Add an option to use SFS without writing each best model on the disk.

## [1.6.4] - 2019-10-29
### added
- Move some functions from vector_tools to raster_tools, functions are anyway still available from vector_tools
### changed
- learnAndPredict manages int value for cross-validation by using RandomStratifiedKFold
- Enhance blocksize management for rasterMath
- Move command line code in _cli folder

## [1.6.3] - 2019-10-14
### changed
- Improvements of rasterMath
	- customBlockSize defines now the same block size for window reading and for the output
	- add seed parameter (to set a random generator) in getRandomBlock()
	- add getRasterParameters() and customRasterParameters() function.

## [1.6.2] - 2019-10-12
### changed
- update rasterMath to generate by default a 256*256 raster block size.
- update rasterMath to prevent bug if user has osgeo/gdal<2.1.
- prevent bug when in rasterMath if processor has only 1 core.
- minor fixes and doc update
