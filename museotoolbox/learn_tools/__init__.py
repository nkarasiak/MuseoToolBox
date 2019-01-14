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
The :mod:`museotoolbox.learn_tools` module gathers learn and predict functions.
"""
from __future__ import absolute_import, print_function
from joblib import Parallel, delayed
import os
import numpy as np
from sklearn import metrics
from sklearn.base import clone
from sklearn import warnings



class learnAndPredict:
    def __init__(self, n_jobs=1, verbose=False):
        """
        learnAndPredict class ease the way to learn a model via an array or a raster using Scikit-Learn algorithm.
        After learning a model via learnFromVector() or learnFromRaster(), you can predict via predictRaster() or predictArray().

        Parameters
        ----------
        n_jobs : int, default 1.
            Number of cores to be used by ``sklearn`` in grid-search.
        verbose : boolean, or int.

        Examples
        --------
        >>> import museotoolbox as mtb
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> raster,vector = mtb.datasets.getHistoricalMap()
        >>> RS50 = mtb.crossValidation.RandomCV(valid_size=0.5,n_splits=10,
                random_state=12,verbose=False)
        >>> classifier = RandomForestClassifier()
        >>> LAP = mtb.learn_tools.learnAndPredict()
        >>> LAP.learnFromRaster(raster,vector,inField='class',cv=RS50,
                    classifier=classifier,param_grid=dict(n_estimators=[100,200]))
        Fitting 10 folds for each of 2 candidates, totalling 20 fits
        best n_estimators : 200
        >>> for kappa in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
            print(kappa)
        [0.94635897652909906]
        [0.93926877916972007]
        [0.9424138426326939]
        [0.9439809301441302]
        [0.94286057027982639]
        [0.94247415327533202]
        [0.94190539222286984]
        [0.94625949356904848]
        [0.94642164578108168]
        [0.9395504758785389]
        >>> LAP.predictRaster(raster,'/tmp/classification.tif')
        Prediction...  [##################......................]45%
        Prediction...  [####################################....]90%
        Saved /tmp/classification.tif using function predictArray
        """
        self.n_jobs = n_jobs
        self.verbose = verbose
        if self.verbose is False:
            warnings.filterwarnings("ignore")
        self.scale = False
        self.CV = False
        self.cloneModel = False

    def scaleX(self, X=None):
        """
        Scale X data using StandardScaler from ``sklearn``.
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
            E.g. RandomForestClassifier() got from ``from sklearn.ensemble import RandomForestClassifier``
        param_grid : None, else dict.
            param_grid for the grid_search. E.g. for RandomForestClassifier : ``param_grid=dict(n_estimators=[10,100],max_features=[1,3])``
        scale : Bool, default False.
            If True, will standardize features.
        scale : Bool, default False.
            If True, will standardize features.
        cv : Cross-Validation or None.
            if cv, choose one from vector_tools.samplingMethods and generate it via vector_tools.sampleSelection().
        """
        self.classifier = classifier
        self.param_grid = param_grid
        self.y = y
        self.group = group

        if cv is not None and self.param_grid is None:
            raise Exception(
                'Please specify a param_grid if you use a cross-validation method')
        self.X = X
        
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
            cv)

    def learnFromRaster(
            self,
            inRaster,
            inVector,
            inField,
            classifier,
            inGroup=False,
            param_grid=None,
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
            E.g. RandomForestClassifier() got from ``from sklearn.ensemble import RandomForestClassifier``
        param_grid : None, else dict.
            param_grid for the grid_search. E.g. for RandomForestClassifier : ``param_grid=dict(n_estimators=[10,100],max_features=[1,3])``
        scale : Bool, default False.
            If True, will standardize features.
        cv : Cross-Validation or None.
            if cv, choose one from vector_tools.samplingMethods and generate it via vector_tools.sampleSelection().
        """
        from ..raster_tools import getSamplesFromROI

        self.classifier = classifier
        self.param_grid = param_grid

        if cv is not False and self.param_grid is None:
            raise Exception(
                'Please specify a param_grid if you use a cross-validation method')

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
            cv)

    def __learn__(self, X, y, groups, classifier,
                  param_grid, cv):
        
        if cv is not False:
            self.CV = []
            for tr, vl in (cv for cv in cv.split(
                    X, y, groups) if cv is not None):
                self.CV.append((tr, vl))
        else:
            self.CV = cv  
            
        from sklearn.model_selection import GridSearchCV

        if isinstance(param_grid, dict):
            grid = GridSearchCV(
                self.classifier,
                param_grid=param_grid,
                cv=cv,
                n_jobs=self.n_jobs,
                verbose=self.verbose + 1)
            grid.fit(X, y, groups)
            self.model = grid.best_estimator_
            self.cloneModel = clone(self.model)
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
            path += '.npz'

        np.savez_compressed(path, LAP=self.__dict__)

        return path

    def loadModel(self, path, onlySKmodel=False):
        """
        Load model previously saved with learnAndPredict.saveModel(path)

        Parameters
        ----------
        path : str.
            If path ends with npy, perfects, else will add '.npy' after your fileName.
        """
        if onlySKmodel:
            print('FutureWarning : From museotoolbox 1.0, loading only SKlearn model will not be available anymore.')
            if not path.endswith('.npy'):
                path += '.npy'

            self.model = np.load(path)

            if type(self.model[1]).__name__ == 'StandardScaler':
                self.scale = True
                self.model, self.scaler = np.load(path)
            else:
                self.scale = False

        else:
            if not path.endswith('npz'):
                path += '.npz'
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

        from ..raster_tools import rasterMath, getGdalDTFromMinMaxValues
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

    def saveCMFromCV(self, savePath, prefix='', header=True, n_jobs=1):
        """
        Save each confusion matrix (csv format) from cross-validation.

        For each matrix, will save as header :

        - The number of training samples per class,
        - The F1-score per class,
        - Overall Accuracy,
        - Kappa.

        Example of confusion matrix saved as csv :

        +-------------------------+------------+
        | # Training samples : 90,80           |
        +-------------------------+------------+
        | # F1 : 91.89,90.32                   |
        +-------------------------+------------+
        | # OA : 91.18                         |
        +-------------------------+------------+
        | # Kappa : 82.23                      |
        +-------------------------+------------+
        |           85            |     5      |
        +-------------------------+------------+
        |           10            |     70     |
        +-------------------------+------------+

        - **In X (columns)** : prediction (95 predicted labels for class 1).
        - **In Y (lines)** : reference (90 labels from class 1).

        Parameters
        ----------
        savePath : str.
            The path where to save the different csv.
            If not exists, will be created
        prefix : str, default ''.
            If prefix, will add this prefix before the csv name (i.e. 0.csv)
        header : boolean, default True.
            If True, will save F1, OA, Kappa and number of training samples.
            If False, will only save confusion matrix

        Returns
        -------
        None

        Examples
        --------
        After having learned with :mod:`museotoolbox.learn_tools.learnAndPredict` :

        >>> LAP.saveCMFromCV('/tmp/testMTB/',prefix='RS50_')
        [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
        [Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed:    3.4s finished
        >>> np.loadtxt('/tmp/testMTB/RS50_0.csv')
        array([[85,  5],
        [10, 70]])
        """
        def __computeStatsPerCV(statsidx, trvl, savePath, prefix, header):
            outFile = savePath + '/' + prefix + str(statsidx) + '.csv'
            dictStats = self.__getStatsFromCVidx(
                statsidx, trvl, True, header, header, header, header)

            if header:
                np_header = 'Training samples : ' + ','.join(str(tr) for tr in dictStats['nTrain']) +\
                    '\nF1 : ' + ','.join(str(np.round(f * 100, 2)) for f in dictStats['F1']) +\
                    '\nOA : {}'.format(np.round(dictStats['OA'] * 100), 2) +\
                    '\nKappa : {}'.format(
                        np.round(dictStats['kappa'] * 100), 2)
            else:
                np_header = ''

            np.savetxt(
                outFile,
                dictStats['confusionMatrix'],
                header=np_header,
                fmt='%0.d')

        if not os.path.exists(savePath):
            os.makedirs(savePath)

        Parallel(n_jobs=n_jobs,
                 verbose=self.verbose + 1)(delayed(__computeStatsPerCV)(statsidx,
                                                                        trvl,
                                                                        savePath,
                                                                        prefix,
                                                                        header) for statsidx,
                                           trvl in enumerate(self.CV))

    def __getStatsFromCVidx(self, statsidx, trvl, confusionMatrix=True,
                            kappa=False, OA=False, F1=False, nTrain=False):
        """
        Compute stats per each fold
        """
        X_train, X_test = self.X[trvl[0]], self.X[trvl[1]]
        Y_train, Y_test = self.y[trvl[0]], self.y[trvl[1]]

        self.cloneModel.fit(X_train, Y_train)

        X_pred = self.cloneModel.predict(X_test)

        accuracies = {}
        if confusionMatrix:
            accuracies['confusionMatrix'] = metrics.confusion_matrix(
                Y_test, X_pred)
        if kappa:
            accuracies['kappa'] = metrics.cohen_kappa_score(Y_test, X_pred)
        if OA:
            accuracies['OA'] = metrics.accuracy_score(Y_test, X_pred)
        if F1:
            accuracies['F1'] = metrics.f1_score(Y_test, X_pred, average=None)
        if nTrain:
            accuracies['nTrain'] = np.unique(Y_train, return_counts=True)[1]

        return accuracies

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
        Accuracies : dict
            A dictionary of each statistic asked.

        Examples
        --------
        After having learned with :mod:`museotoolbox.learn_tools.learnAndPredict` :

        >>> for stats in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
        >>> stats['kappa']
        0.942560083148
        0.94227598585
        0.942560083148
        ...
        """
        def __computeStatsPerCV(statsidx, trvl, **kwargs):
            dictStats = self.__getStatsFromCVidx(statsidx, trvl, **kwargs)
            return dictStats
        if self.CV is False:
            raise Exception(
                'You must have learnt with a Cross-Validation')
        else:
            statsCV = Parallel(
                n_jobs=-1,
                verbose=self.verbose)(
                delayed(__computeStatsPerCV)(
                    statsidx,
                    trvl,
                    confusionMatrix=confusionMatrix,
                    kappa=kappa,
                    OA=OA,
                    F1=F1,
                    nTrain=nTrain) for statsidx,
                trvl in enumerate(
                    self.CV))
            return statsCV
