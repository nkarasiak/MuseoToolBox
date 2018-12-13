#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# ___  ___                       _____           _______
# |  \/  |                      |_   _|         | | ___ \
# | .  . |_   _ ___  ___  ___     | | ___   ___ | | |_/ / _____  __
# | |\/| | | | / __|/ _ \/ _ \    | |/ _ \ / _ \| | ___ \/ _ \ \/ /
# | |  | | |_| \__ \  __/ (_) |   | | (_) | (_) | | |_/ / (_) >  <
# \_|  |_/\__,_|___/\___|\___/    \_/\___/ \___/|_\____/ \___/_/\_\
#
# @author:  Nicolas Karasiak
# @site:    www.karasiak.net
# @git:     www.github.com/lennepkade/MuseoToolBox
# =============================================================================
"""
The :mod:`museotoolbox.learnTools` module gathers learn and predict functions.
"""
from __future__ import absolute_import, print_function
import numpy as np
import pickle

class learnAndPredict:
    def __init__(self, n_jobs=1, verbose=False):
        """
        learnAndPredict class ease the way to learn a model via an array or a raster using Scikit-Learn algorithm.
        After learning a model via learnFromVector() or learnFromRaster(), you can predict via predictRaster() or predictArray().


        Parameters
        ----------
        n_jobs : int, default 1.
            Number of cores to be used by sklearn in grid-search.
        """
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.scale = False
        self.CV = False

    def scaleX(self, X=None):
        """
        Scale X data using StandardScaler from sklear.
        If X is None, initialize StandardScaler.

        Parameters
        ----------
         X : arr, or None.
             The array to scale the data from.

        """
        from sklearn.preprocessing import StandardScaler
        if X is None:
            self.scaler = StandardScaler()
        else:
            self.scaler.fit(X)
            Xt = self.scaler.transform(X)
            return Xt

    def learnFromVector(
            self,
            X,
            y,
            group=None,
            classifier=None,
            param_grid=None,
            outStatsFromCV=True,
            scale=False,
            cv=False):
        """
        learn Model from vector/array.

        Parameters
        ----------
        X : array.
            Array with values of each label variable.
        Y : array.
            Array with labels only.
        classifier : class from scikit-learn.
            E.g. RandomForestClassifier() got from 'from sklearn.ensemble import RandomForestClassifier'
        param_grid : None, else dict.
            param_grid for the grid_search. E.g. for RandomForestClassifier : param_grid=dict(n_estimators=[10,100],max_features=[1,3])
        outStatsFromCv : bool, default True.
            If True, getStatsFromCV() will be available to keep statistics (confusion matrix, OA, Kappa, F1)
        scale : Bool, default False.
            If True, will standardize features.
        scale : Bool, default False.
            If True, will standardize features.
        cv : Cross-Validation or None.
            if cv, choose one from vectorTools.samplingMethods and generate it via vectorTools.sampleSelection().
        """
        self.classifier = classifier
        self.param_grid = param_grid
        self.y = y
        self.group = group

        if cv is not None and self.param_grid is None:
            raise Exception(
                'Please specify a param_grid if you use a cross-validation method')
        self.X = X
        self.CV = cv
        if scale:
            self.scale = True
            self.scaleX()
            self.X = self.scaleX(X)

        self.__learn__(
            self.X,
            self.y,
            self.group,
            classifier,
            param_grid,
            outStatsFromCV,
            cv)

    def learnFromRaster(
            self,
            inRaster,
            inVector,
            inField,
            classifier,
            inGroup=False,
            param_grid=None,
            outStatsFromCV=True,
            scale=False,
            cv=False):
        """
        learn Model from raster.

        Parameters
        ----------
        inRaster : str.
            Path of the raster file.
        inVector : str.
            Path of the vector file.
        inField : str.
            Field name containing the label to predict.
        classifier : class from scikit-learn.
            E.g. RandomForestClassifier() got from 'from sklearn.ensemble import RandomForestClassifier'
        param_grid : None, else dict.
            param_grid for the grid_search. E.g. for RandomForestClassifier : param_grid=dict(n_estimators=[10,100],max_features=[1,3])
        outStatsFromCv : bool, default True.
            If True, getStatsFromCV() will be available to keep statistics (confusion matrix, OA, Kappa, F1)
        scale : Bool, default False.
            If True, will standardize features.
        cv : Cross-Validation or None.
            if cv, choose one from vectorTools.samplingMethods and generate it via vectorTools.sampleSelection().
        """
        from ..rasterTools import getSamplesFromROI

        self.classifier = classifier
        self.param_grid = param_grid

        if cv is not False and self.param_grid is None:
            raise Exception(
                'Please specify a param_grid if you use a cross-validation method')
        self.CV = cv

        if inGroup is False:
            group = None
            X, y = getSamplesFromROI(
                inRaster, inVector, inField, verbose=self.verbose)
        else:
            X, y, group = getSamplesFromROI(
                inRaster, inVector, inField, inGroup, verbose=self.verbose)

        self.y = y
        self.X = X
        self.group = group

        if scale:
            self.scale = True
            self.scaleX()
            self.X = self.scaleX(X)
        self.Xpredict = False

        self.__learn__(
            self.X,
            self.y,
            self.group,
            classifier,
            param_grid,
            outStatsFromCV,
            cv)

    def __learn__(self, X, y, groups, classifier,
                  param_grid, outStatsFromCV, cv):
        self.outStatsFromCV = outStatsFromCV
        from sklearn.model_selection import GridSearchCV
        if outStatsFromCV is True and cv is not None:
            self.CV = []
            for tr, vl in (cv for cv in cv.split(
                    X, y, groups) if cv is not None):
                self.CV.append((tr, vl))
        else:
            self.CV = cv

        if isinstance(param_grid, dict):
            grid = GridSearchCV(
                self.classifier,
                param_grid=param_grid,
                cv=self.CV,
                n_jobs=self.n_jobs,
                verbose=self.verbose + 1)
            grid.fit(X, y, groups)
            self.model = grid.best_estimator_
            self.model.fit(X, y, groups)
            for key in self.param_grid.keys():
                message = 'best ' + key + ' : ' + str(grid.best_params_[key])
                print(message)

        else:
            self.model = self.classifier.fit(X=X, y=y)

    def saveModel(self, path):
        """
        Save model 'myModel.npz' to be loaded later via learnAndPredict.loadModel(path)

        Parameters
        ----------
        path : str.
            If path ends with npz, perfects, else will add '.npz' after your fileName.
            
        Returns
        -------
        path : str.
            Path and filename with mtb extension.
        """
        if not path.endswith('npz'):
            path+='.npz'

        np.savez_compressed(path,LAP=self.__dict__)
        
        return path
    
    def loadModel(self, path):
        """
        Load model previously saved with learnAndPredict.saveModel(path)

        Parameters
        ----------
        path : str.
            If path ends with npy, perfects, else will add '.npy' after your fileName.
        """
        model = np.load(path)
        self.__dict__.update(model['LAP'].tolist())

    def predictArray(self, X):
        """
        Predict label from array.

        Parameters
        ----------
        X : array.
            The array to predict. Must have the same number of bands of the initial array/raster.
        """
        if self.scale:
            X = self.scaler.transform(X)
        Xpredict = self.model.predict(X)
        return Xpredict

    def predictConfidencePerClass(self, X):
        """
        Predict label from array.

        Parameters
        ----------
        X : array.
            The array to predict proba. Must have the same number of bands of the initial array/raster.


        Returns
        ----------
        Xpredict : array.
            The probability from 0 to 100.
        """
        if self.scale:
            X = self.scaler.transform(X)
        Xpredict = self.model.predict_proba(X) * 100
        self.Xpredict = Xpredict
        return Xpredict

    def predictConfidenceOfPredictedClass(self, X):
        """
        Predict label from array.

        Parameters
        ----------
        X : array.
            The array to predict proba. Must have the same number of bands of the initial array/raster.

        Returns
        ----------
        Xpredict : array.
            The probability from 0 to 100.
        """
        if hasattr(self, 'Xpredict'):
            Xpredict = np.amax(self.Xpredict, axis=1)
        else:
            if self.scale:
                X = self.scaler.transform(X)
            Xpredict = np.amax(self.model.predict_proba(X) * 100, axis=1)
        return Xpredict

    def predictRaster(
            self,
            inRaster,
            outRaster,
            confidencePerClass=False,
            confidence=False,
            inMaskRaster=False,
            outNoData=0):
        """
        Predict label from raster using previous learned model.
        This function will call self.predictArray(X).

        Parameters
        ----------
        inRaster : str
            Path of the raster used for prediction.
        outRaster : str
            Path of the prediction raster to save.
        outConfidence : str
            Path of the max confidence from all classes raster to save.
        outCOnfidencePerClass : str
            Path of the confidence raster per class to be saved.
        inMaskRaster : str, default False.
            Path of the raster where 0 is mask and value above are no mask.
        outGdalDT : int, defaut 1.
            1 is for gdal.GDT_Byte, 2 for gdal.GDT_UInt16, 3 is for gdal.GDT_Int16...
        outNoData : int, default 0.
            Value of no data for the outRaster.
        """

        from ..rasterTools import rasterMath
        from ..rasterTools import getGdalDTFromMinMaxValues
        rM = rasterMath(inRaster, inMaskRaster, 'Prediction... ')

        gdalDT = getGdalDTFromMinMaxValues(np.amax(self.model.classes_))

        rM.addFunction(
            self.predictArray,
            outRaster,
            outNBand=1,
            outGdalDT=gdalDT,
            outNoData=outNoData)

        noDataConfidence = -9999

        if confidencePerClass:
            rM.addFunction(
                self.predictConfidencePerClass,
                confidencePerClass,
                outNBand=False,
                outGdalDT=getGdalDTFromMinMaxValues(100, noDataConfidence),
                outNoData=noDataConfidence)

        if confidence:
            rM.addFunction(
                self.predictConfidenceOfPredictedClass,
                confidence,
                outNBand=1,
                outGdalDT=getGdalDTFromMinMaxValues(100, noDataConfidence),
                outNoData=noDataConfidence)
        rM.run()

    def getStatsFromCV(
            self,
            confusionMatrix=True,
            kappa=False,
            OA=False,
            F1=False,
            nTrain=False):
        """
        Extract statistics from the Cross-Validation.
        If Cross-Validation is 5-fold, getStatsFromCV will return 5 confusion matrix, 5 kappas...

        Parameters
        -----------
        confusionMatrix : bool, default True.
            If True, will return first the Confusion Matrix.
        kappa : bool, default False.
            If True, will return in kappa.
        OA : bool, default False.
            If True, will return Overall Accuracy/
        F1 : bool, default False.
            If True, will return F1 Score per class.
        nTrain : bool, default False.
            If True, will return numbe of train samples ordered asc. per label.

        Returns
        -------
        Statistics : List of statistics with at least the confusion matrix.
        """
        if self.CV is False:
            raise Exception(
                'You must have learnt with a Cross-Validation')
        else:
            # ,statsFromConfusionMatrix,
            from ..stats import computeConfusionMatrix as computeStats

            for train_index, test_index in self.CV:
                results = []
                X_train, X_test = self.X[train_index], self.X[test_index]
                Y_train, Y_test = self.y[train_index], self.y[test_index]

                self.model.fit(X_train, Y_train)
                X_pred = self.model.predict(X_test)
                cmObject = computeStats(
                    Y_test, X_pred, kappa=kappa, OA=OA, F1=F1)
                if confusionMatrix:
                    results.append(cmObject.confusion_matrix)
                if kappa:
                    results.append(cmObject.Kappa)
                if OA:
                    results.append(cmObject.OA)
                if F1:
                    results.append(cmObject.F1)
                if nTrain:
                    results.append(np.unique(Y_train, return_counts=True)[1])

                yield results
