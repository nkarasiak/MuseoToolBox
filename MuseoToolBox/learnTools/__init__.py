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
from __future__ import absolute_import, print_function
import numpy as np

class learnAndPredict:
    def __init__(self, n_jobs=1):
        """
        learnAndPredict class ease the way to learn a model via an array or a raster using Scikit-Learn algorithm.
        After learning a model via learnFromVector() or learnFromRaster(), you can predict via predictFromRaster() or predictFromArray().


        Parameters
        ----------
        n_jobs : int, default 1.
            Number of cores to be used by sklearn in grid-search.
        """
        self.n_jobs = n_jobs

        self.scale = False

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
            Y,
            classifier,
            param_grid=None,
            outStatsFromCV=True,
            scale=False,
            cv=None):
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
        self.Y = Y.reshape(-1,1)

        self.X = X
        if scale:
            self.scale = True
            self.scaleX()
            self.X = self.scaleX(X)

        self.__learn__(self.X, self.Y, outStatsFromCV, cv)

    def learnFromRaster(
            self,
            inRaster,
            inVector,
            inField,
            classifier,
            param_grid=None,
            outStatsFromCV=True,
            scale=False,
            cv=None):
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

        X, Y = getSamplesFromROI(inRaster, inVector, inField)
        self.Y = Y

        self.X = X
        if scale:
            self.scale = True
            self.scaleX()
            self.X = self.scaleX(X)
        self.Xpredict = False

        self.__learn__(X, Y, outStatsFromCV, cv)

    def __learn__(self, X, Y, outStatsFromCV, cv):
        self.outStatsFromCV = outStatsFromCV
        from sklearn.model_selection import GridSearchCV
        if cv is None:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=3)

        if outStatsFromCV is True:
            self.CV = []
            if hasattr(
                    cv, 'split'):  # if cv is from sklearn, need to fit to generate tr/vl
                try:
                    cv = cv.split(X, Y)
                except BaseException:
                    pass
            for tr, vl in cv:
                self.CV.append((tr, vl))
        else:
            self.CV = cv

        if isinstance(self.param_grid, dict):
            grid = GridSearchCV(
                self.classifier,
                param_grid=self.param_grid,
                cv=self.CV,
                n_jobs=self.n_jobs)
            grid.fit(X, Y)
            self.model = grid.best_estimator_
            self.model.fit(X, Y)
            for key in self.param_grid.keys():
                message = 'best ' + key + ' : ' + str(grid.best_params_[key])
                print(message)

        else:
            self.model = self.classifier.fit(self,X=X,y=Y.reshape(-1,1))

    def saveModel(self, path):
        """
        Save model to be load later via learnAndPredict.loadModel(path)

        Parameters
        ----------
        path : str.
            If path ends with npy, perfects, else will add '.npy' after your fileName.
        """
        if not path.endswith('.npy'):
            path += '.npy'

        if self.scale:
            np.save(path, [self.model, self.scaler])
        else:
            np.save(path, self.model)

    def loadModel(self, path):
        """
        Load model previously saved with learnAndPredict.saveModel(path)

        Parameters
        ----------
        path : str.
            If path ends with npy, perfects, else will add '.npy' after your fileName.
        """
        if not path.endswith('.npy'):
            path += '.npy'

        if self.scale:
            self.scale = True
            self.model, self.scaler = np.load(path)
        else:
            self.scale = False
            self.model = np.load(path)

        # if not self.scale but scale was used in a previous call of
        # learnAndPredict
        if type(self.model[1]).__name__ == 'StandardScaler':
            self.scale = True
            self.model, self.scaler = np.load(path)

    def predictFromArray(self, X):
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

    def predictConfidencePerClassFromArray(self, X):
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

    def predictConfidenceOfPredictedClassFromArray(self, X):
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

    def predictFromRaster(
            self,
            inRaster,
            outRaster,
            outConfidencePerClass=False,
            outConfidence=False,
            inMaskRaster=False,
            outNoData=0):
        """
        Predict label from raster using previous learned model.
        This function will call self.predictFromArray(X).

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
        gdalDT = getGdalDTFromMinMaxValues(int(np.amax(np.unique(self.Y))))
        rM.addFunction(
            self.predictFromArray,
            outRaster,
            1,
            gdalDT,
            outNoData=0)
        if outConfidencePerClass:
            rM.addFunction(
                self.predictConfidencePerClassFromArray,
                outConfidencePerClass,
                outNBand=False,
                outGdalDT=3,
                outNoData=-9999)
        if outConfidence:
            rM.addFunction(
                self.predictConfidenceOfPredictedClassFromArray,
                outConfidence,
                outNBand=False,
                outGdalDT=3,
                outNoData=-9999)
        rM.run()

    def getStatsFromCV(
            self,
            confusionMatrix=True,
            kappa=False,
            OA=False,
            F1=False):
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
            
        Returns
        -------
        Statistics : List of statistics with at least the confusion matrix.
        """
        if self.outStatsFromCV is False:
            raise Exception(
                'outStatsFromCV in fromRaster or fromVector must be True')
        else:
            # ,statsFromConfusionMatrix,
            from ..stats.statsFromConfusionMatrix import confusionMatrix
            CM = []
            kappas = []
            OAs = []
            F1s = []
            for train_index, test_index in self.CV:
                X_train, X_test = self.X[train_index], self.X[test_index]
                Y_train, Y_test = self.Y[train_index], self.Y[test_index]

                self.model.fit(X_train, Y_train)
                X_pred = self.model.predict(X_test)
                cmObject = confusionMatrix(
                    Y_test, X_pred, kappa=kappa, OA=OA, F1=F1)
                cm = cmObject.confusion_matrix
                CM.append([cm])
                if kappa:
                    kappas.append(cmObject.Kappa)
                if OA:
                    OAs.append(cmObject.OA)
                if F1:
                    F1s.append(cmObject.F1)

            toReturn = []
            if confusionMatrix:
                toReturn.append(CM)
            if kappa is True:
                toReturn.append(kappas)
            if OA is True:
                toReturn.append(OAs)
            if F1 is True:
                toReturn.append(F1s)
            return toReturn
