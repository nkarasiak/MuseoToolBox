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
# @git:     www.github.com/nkarasiak/MuseoToolBox
# =============================================================================
"""
The :mod:`museotoolbox.learn_tools` module gathers machine learning classes.
"""
from joblib import Parallel, delayed
import os
import numpy as np
from sklearn import metrics
from sklearn.base import clone
from sklearn import warnings
from ..raster_tools import rasterMath, _getGdalDTFromMinMaxValues, _convertGdalAndNumpyDataType
from ..internal_tools import progressBar


class learnAndPredict:
    def __init__(self, n_jobs=1, verbose=False):
        """
        learnAndPredict  ease the way to learn a model via an array or a raster using Scikit-Learn algorithm.
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
        >>> raster,vector = mtb.datasets.historicalMap()
        >>> RS50 = mtb.cross_validation.RandomStratifiedKFold(n_splits=2,n_repeats=5,
                random_state=12,verbose=False)
        >>> classifier = RandomForestClassifier()
        >>> LAP = mtb.learn_tools.learnAndPredict(verbose=True)
        >>> LAP.learnFromRaster(raster,vector,inField='class',cv=RS50,
                    classifier=classifier,param_grid=dict(n_estimators=[100,200]))
        Reading raster values...  [########################################]100%
        Fitting 10 folds for each of 2 candidates, totalling 20 fits
        best score : 0.966244859222
        best n_estimators : 200
        >>> for kappa in LAP.getStatsFromCV(confusionMatrix=False,kappa=True):
            print(kappa)
        [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
        {'kappa': 0.94145803865870303}
        {'kappa': 0.94275572196698443}
        {'kappa': 0.94566553229314054}
        {'kappa': 0.94210064101370472}
        {'kappa': 0.94566137634353153}
        {'kappa': 0.94085890364956737}
        {'kappa': 0.94136385707385184}
        {'kappa': 0.9383201352573155}
        {'kappa': 0.93887726891376944}
        {'kappa': 0.94450020549861891}
        [Parallel(n_jobs=-1)]: Done  10 out of  10 | elapsed:    8.7s finished
        >>> LAP.predictRaster(raster,'/tmp/classification.tif')
        Total number of blocks : 15
        Prediction...  [########################################]100%
        Saved /tmp/classification.tif using function predictArray
        """
        self.n_jobs = n_jobs
        self.verbose = verbose
        if self.verbose is False:
            warnings.filterwarnings("ignore")
        self.standardize = False
        self.standardized = False
        self._x_is_customized = False
        self.xKwargs = {}
        self.CV = False
        self.cloneModel = False

    def standardizeX(self, X=None,need_transformation=False):
        """
        Scale X data using StandardScaler from ``sklearn``.
        If X is None, initialize StandardScaler.

        Parameters
        ----------
         X : arr, or None.
             The array to scale the data from.
        need_transformation : bool, default True
            If you used function to transform your array.

        """
        from sklearn.preprocessing import StandardScaler

        try:
            self.StandardScaler
        except BaseException:
            self.StandardScaler = StandardScaler()

        if X is not None:
            if need_transformation:    
                if self._x_is_customized:
                    X = self.xFunction(X, **self.xKwargs)
            if self.standardized is False:
                self.StandardScaler.fit(X)
                self.standardized = True

            Xt = self.StandardScaler.transform(X)

            return Xt

    def learnFromVector(
            self,
            X,
            y,
            classifier=None,
            group=None,
            param_grid=None,
            standardize=True,
            cv=None,
            scoring='accuracy',
            refit=True,
            **gridSearchCVParams):
        """
        learn Model from vector/array.

        Parameters
        ----------
        X : array.
            Array with values of each label variable.
        y : array.
            Array with labels only.
        classifier : class from scikit-learn.
            E.g. RandomForestClassifier() got from ``from sklearn.ensemble import RandomForestClassifier``
        group : str or False.
            If you use a cross-validation which needs group-splitting.
        param_grid : None, else dict.
            param_grid for the grid_search. E.g. for RandomForestClassifier : ``param_grid=dict(n_estimators=[10,100],max_features=[1,3])``
        strandardize : Bool, default True.
            If True, will standardize features by removing the mean and scaling to unit variance.
        cv : Cross-Validation or int or None. Default None.
            if cv, choose one from cross_validation.
            if int, uses :class:`museotoolbox.cross_validation.RandomStratifiedKFold` with K = the int value.
        """
        self.classifier = classifier
        self.param_grid = param_grid
        self.y = y
        self.group = group

        if self._x_is_customized:
            X = self.xFunction(X, **self.xKwargs)

        self.X = X

        if standardize:
            self.standardize = True
            self.standardizeX()
            self.X = self.standardizeX(X,need_transformation=False)

        self.__learn__(
            self.X,
            self.y,
            self.group,
            classifier,
            param_grid,
            cv,
            scoring,
            refit,
            **gridSearchCVParams)

    def learnFromRaster(
            self,
            inRaster,
            inVector,
            inField,
            classifier=None,
            group=None,            
            param_grid=None,
            standardize=True,
            cv=None,
            scoring='accuracy',
            refit=True,
            **gridSearchCVParams):
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
        group : str or False.
            If you use a cross-validation which needs group-splitting.
        param_grid : None, else dict.
            param_grid for the grid_search. E.g. for RandomForestClassifier : ``param_grid=dict(n_estimators=[10,100],max_features=[1,3])``
        standardize : Bool, default True.
            If True, will standardize features by removing the mean and scaling to unit variance.
        cv : Cross-Validation or int or None. Default 2.
            if cv, choose one from cross_validation.
            if int, uses :class:`museotoolbox.cross_validation.RandomStratifiedKFold` with K = the int value.
        """
        from ..raster_tools import getSamplesFromROI

        self.classifier = classifier
        self.param_grid = param_grid

        if group is False or group is None:
            group = None
            X, y = getSamplesFromROI(
                inRaster, inVector, inField, verbose=self.verbose)
        else:
            X, y, group = getSamplesFromROI(
                inRaster, inVector, inField, group, verbose=self.verbose)

        self.y = y
        self.X = X
        self.group = group

        if self._x_is_customized is True:
            self.X = self.xFunction(X, **self.xKwargs)
        if standardize:
            self.standardize = True
            self.standardizeX()
            self.X = self.standardizeX(self.X,need_transformation=False)
        self.Xpredict = False

        self.__learn__(
            self.X,
            self.y,
            self.group,
            classifier,
            param_grid,
            cv,
            scoring,
            refit,
            **gridSearchCVParams)

    def __learn__(self, X, y, groups, classifier,
                  param_grid, cv, scoring='accuracy', refit=True, **gridSearchCVParams):
        
        
        if isinstance(cv, int) and cv != False:
            from ..cross_validation import RandomStratifiedKFold
            cv = RandomStratifiedKFold(n_splits=cv)
        if cv is False or cv is None and isinstance(param_grid, dict):
            raise ValueError('You need to select a cross-validation method to use the param_grid for the gridSearchCV.')
        if cv is not None and cv is not False:
            self.CV = []
            for tr, vl in (cv for cv in cv.split(
                    X, y, groups) if cv is not None):
                self.CV.append((tr, vl))
        elif cv is not False:
                self.CV = cv
            

        from sklearn.model_selection import GridSearchCV

        if isinstance(param_grid, dict):
            self.model = GridSearchCV(
                self.classifier,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                refit=refit,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                **gridSearchCVParams)
            self.model.fit(X, y, groups)
            # self.model = self.gS.best_estimator_
            self.cloneModel = clone(self.model.best_estimator_)
            #self.model.fit(X, y, groups)
            if self.verbose:
                print('best score : ' + str(self.model.best_score_))
                for key in self.param_grid.keys():
                    message = 'best ' + key + ' : ' + \
                        str(self.model.best_params_[key])
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
                self.standardize = True
                self.model, self.StandardScaler = np.load(path)
            else:
                self.standardize = False

        else:
            if not path.endswith('npz'):
                path += '.npz'
            model = np.load(path, allow_pickle=True)
            self.__dict__.update(model['LAP'].tolist())
            if hasattr(self, 'scale'):
                if self.scale is True:
                    # for retro-compatibility with museotoolbox < 1.0
                    self.standardize, self.StandardScaler = self.scale, self.scaler
                    del self.scaler, self.scale

    def __convertX(self, X):
        if self._x_is_customized is True:
            X = self.xFunction(X, **self.xKwargs)

        if self.standardize:
            if np.ma.is_masked(X):
                tmpMask = X.mask[:, 0]

            X = self.StandardScaler.transform(X)

            if np.ma.is_masked(X):
                tmpMask = np.repeat(tmpMask.reshape(-1, 1),
                                    X.shape[-1], axis=1)
                X = np.ma.masked_array(X, tmpMask)

        return X

    def predictArray(self, X):
        """
        Predict label from array.

        Parameters
        ----------
        X : array.
            The array to predict. Must have the same number of bands of the initial array/raster.
        **kwargs :
            Xfunction : a custom function to modify directly the array from the raster.
        """

        X = self.__convertX(X)
        self.Xpredict = self.model.predict(X)

        return self.Xpredict

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
        X = self.__convertX(X)

        Xpredict_proba = self.model.predict_proba(X) * 100
        if Xpredict_proba.ndim == 1:
            Xpredict_proba = Xpredict_proba.reshape(-1, 1)
        # share prediction in class in case of confidence for only predicted
        # class
        self.Xpredict_proba = Xpredict_proba
        return Xpredict_proba

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
        if hasattr(self, 'Xpredict_proba'):
            Xpredict_proba = np.amax(self.Xpredict_proba, axis=1)
        else:
            Xpredict_proba = np.amax(
                self.model.predict_proba(
                    self.__convertX(
                        X)) * 100, axis=1)
        return Xpredict_proba

    def predictRaster(
            self,
            inRaster,
            outRaster,
            confidencePerClass=False,
            confidence=False,
            inMaskRaster=False,
            outNoData=0,
            compress=True):
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
        outConfidencePerClass : str
            Path of the confidence raster per class to be saved.
        inMaskRaster : str, default False.
            Path of the raster where 0 is mask and value above are no mask.
        outNumpyDT : numpy datatype, default will get the datatype according to your maximum class value.
            Get numpy datatype throught : _convertGdalAndNumpyDataType(_getGdalDTFromMinMaxValues(maximumClassValue)))
        outNoData : int, default 0.
            Value of no data for the outRaster.
        """

        rM = rasterMath(inRaster, inMaskRaster, message='Prediction...')

        numpyDT = _convertGdalAndNumpyDataType(
            _getGdalDTFromMinMaxValues(np.amax(self.model.classes_)))

        rM.addFunction(
            self.predictArray,
            outRaster,
            outNBand=1,
            outNumpyDT=numpyDT,
            outNoData=outNoData,
            compress=compress)

        if confidencePerClass:
            rM.addFunction(
                self.predictConfidencePerClass,
                confidencePerClass,
                outNBand=False,
                outNumpyDT=np.int16,
                outNoData=np.iinfo(np.int16).min,
                compress=compress)

        if confidence:
            rM.addFunction(
                self.predictConfidenceOfPredictedClass,
                confidence,
                outNBand=1,
                outNumpyDT=np.int16,
                outNoData=np.iinfo(np.int16).min,
                compress=compress)
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
            If True, will return number of train samples ordered asc. per label.

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
                n_jobs=self.n_jobs,
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

    def customizeX(self, xFunction, **kwargs):
        self._x_is_customized = True
        self.xFunction = xFunction
        self.xKwargs = kwargs


class sequentialFeatureSelection:
    """
    Sequential Feature Selection

    Parameters
    ----------
    classifier : class.
        Classifier from scikit-learn.    
    param_grid : array.
        param_grid for hyperparameters of the classifier.
    cv : int, default 5.
    scoring : str or class, optional
        default is 'accuracy'. See sklearn.metrics.make_scorer from scikit-learn.
    n_comp : int, optional
        The number of component per feature. If 4, each feature has 4 columns.
    verbose : int, optional
        The higher it is the more sequential will show progression.
    """

    def __init__(self, classifier, param_grid, cv=5,
                 scoring='accuracy', n_comp=1, verbose=1):
        # share args
        self.n_comp = n_comp
        self.classifier = classifier
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.verbose = verbose

        self.xFunction = False
        self.xKwargs = False

    def fit(self, X, y, group=None, standardize=True,
            path_to_save_models=False, max_features=False, n_jobs=1,**kwargs):
        """
        Parameters
        ----------
        X : arr
            arr
        y : arr
            Size of X.shape[0].
        group : None, optional
            group for cross-validation
        standardize : optional
            Default True.
        path_to_save_models : str.
            Path to save models.
        max_features : int or bool.
            Default False, if value int.
        n_jobs : int.
            Number of job to compute cross-validation.
        """
        # for retrocompatibily museotoolbox <1.6.5
        if 'pathToSaveCM' in kwargs and path_to_save_models is False:    
            print('Please use path_to_save_models instead of pathToSaveCM')
            self.path_to_save_models = kwargs['pathToSaveCM']
        else:
            self.path_to_save_models = path_to_save_models

        self.X = X
        self.X_ = np.copy(X)
        self.y = y
        self.group = group

        self.models_path_ = []

        if self.xFunction:
            self.X = self.xFunction(X, **self.xKwargs)

        xSize = self.X.shape[1]
        self.n_features = int(xSize / self.n_comp)

        self.max_features = self.n_features

        if max_features is not False:
            if max_features < self.n_features:
                self.max_features = max_features

        totalIter = np.sum(
            [self.n_features - i for i in range(self.max_features)])

        if self.verbose:
            pB = progressBar(totalIter, message='SFFS:')
        self.mask = np.ones(xSize, dtype=bool)

        self.models_, self.best_scores_, self.best_features_ = [[], [], []]
        self.subsets_ = dict()

        for j in range(self.max_features):
            resPerFeatures = list()
            need_learn = True
            # bestScore,LAPs,bestParams,cvResults,models = [[],[],[],[],[]dd]
            n_features_to_test = int(
                self.X[:, self.mask].shape[1] / self.n_comp)
            if self.path_to_save_models :
                all_scores_file = self.path_to_save_models + 'all_scores_{}.csv'.format(j)
                if os.path.exists(all_scores_file):
                    need_learn = False
                    print('Feature {} already computed'.format(j))
                    scores = np.loadtxt(all_scores_file, delimiter=',')
    
                    if scores.ndim == 1:
                        all_scores = [scores[1]]
                        best_candidate = 0
                    else:
                        all_scores = scores[:, 1]
                        best_candidate = np.argmax(scores[:, 1])
                    LAP = learnAndPredict(n_jobs=n_jobs, verbose=self.verbose - 1)
                    LAP.loadModel(self.path_to_save_models + 'model_{}.npz'.format(j))
                    self.models_path_.append(
                        self.path_to_save_models + 'model_{}.npz'.format(j))
                
            if need_learn is True:
                for idx in range(
                        n_features_to_test):  # at each loop, remove best candidate
                    if self.verbose:
                        pB.addPosition()
                    LAP = learnAndPredict(
                        n_jobs=n_jobs, verbose=self.verbose - 1)
                    curX = self.__transformInFit(self.X, idx)
                    if standardize is False:
                        scale = False
                    else:
                        scale = True

                    LAP.learnFromVector(
                        curX,
                        y,
                        group=group,
                        classifier=self.classifier,
                        param_grid=self.param_grid,
                        standardize=scale,
                        scoring=self.scoring,
                        cv=self.cv)

                    resPerFeatures.append(LAP)

                all_scores = [np.amax(LAP.model.best_score_)
                              for LAP in resPerFeatures]
                best_candidate = np.argmax(all_scores)
                # self.__bestLAPs.append(resPerFeatures[best_candidate])
                LAP = resPerFeatures[best_candidate]

                if self.path_to_save_models:
                    if not os.path.exists(os.path.join(self.path_to_save_models, str(j))):
                        os.makedirs(os.path.join(self.path_to_save_models, str(j)))
                    LAP.saveModel(self.path_to_save_models + 'model_{}.npz'.format(j))
                    LAP.saveCMFromCV(
                        os.path.join(
                            self.path_to_save_models,
                            str(j)),
                        n_jobs=n_jobs)
            
                if self.n_comp == 1:
                    bandidx = np.where(self.mask == 1)[0].reshape(-1, 1)
                else:

                    bandidx = np.arange(
                        0, self.mask.shape[0], self.n_comp).reshape(-1, 1)
                    bandidx = np.int32(
                        bandidx[np.in1d(self.mask[bandidx], 1)] / self.n_comp)

                scoreWithIdx = np.hstack((bandidx, np.asarray(
                    all_scores, dtype=np.float32).reshape(-1, 1)))
                if self.path_to_save_models:
                    np.savetxt(all_scores_file, scoreWithIdx, fmt='%0.d,%.4f')
                    self.models_path_.append(self.path_to_save_models + 'model_{}.npz'.format(j))
                else:
                    self.models_.append(resPerFeatures[best_candidate])
                
                # store results
            best_feature_id = int(
                self.__getFeatureId(best_candidate) / self.n_comp)
            self.best_scores_.append(all_scores[best_candidate])
            self.best_features_.append(best_feature_id)

            if self.verbose:
                print(
                    '\nBest feature with %s feature(s) : %s' %
                    (j + 1, best_feature_id))
                print('Best mean score : %s' % np.amax(all_scores))

            self.subsets_[str(j)] = dict(avg_score=np.amax(all_scores),
                                         feature_idx=self.best_features_.copy(),
                                         cv_score=LAP.model.cv_results_,
                                         best_score_=np.amax(all_scores),
                                         best_feature_=best_feature_id)

            self.__maskIdx(best_candidate)

    def predict(self, X, idx):
        """
        Predict in raster using the best features.

        Parameters
        -----------
        X : array.
            The array to predict. Must have the same number of bands of the initial array/raster.
        idx : int.
            The combination (from 0).
        """

        self.__resetMask()

        if self.path_to_save_models  is False:
            LAP = learnAndPredict(n_jobs=1, verbose=self.verbose)
            LAP.loadModel(self.models_path_[idx])
        else:
            LAP = self.models_[idx]
        return LAP.predictArray(self.transform(X, idx=idx))

    def predictBestCombination(
            self, inRaster, outRaster, inMaskRaster=False, confidence=False):
        """
        Predict in raster using the best features.

        Parameters
        -----------
        inRaster : str.
            Path of the raster to predict.
        outRaster : str.
            Path of the image to save (GeoTiff file).
        inMaskRaster : str.
            Path of the raster mask where 0 is mask.
        confidence : False or str. Default False.
            If str, same as outRasterPrefix.
        """

        self.__resetMask()
        self.best_idx_ = np.argmax(self.best_scores_)

        print('Predict with combination ' + str(self.best_idx_))
        
        if self.path_to_save_models  is False:
            LAP = self.models_[self.best_idx_]
        else:
            LAP = learnAndPredict(n_jobs=1, verbose=self.verbose)
            LAP.loadModel(self.models_path_[self.best_idx_])
            
        LAP.customizeX(self.transform, idx=self.best_idx_, customizeX=True)
            
            
        LAP.predictRaster(inRaster, outRaster,
                          confidence=confidence, inMaskRaster=inMaskRaster)

    def _relearnBestModelWithBestParams(
            self, X, y, group=None, standardize=True, n_jobs=1, save_dir=False):

        self.__resetMask()
        self.best_idx_ = np.argmax(self.best_scores_)

        model = self.models_path_[self.best_idx_]
        LAP = learnAndPredict(
            n_jobs=n_jobs, verbose=self.verbose - 1)
        if self.xFunction is not False:
            customizeX = True
        else:
            customizeX = False
        curX = self.transform(X, self.best_idx_, customizeX=customizeX)
        if standardize is False:
            scale = False
        else:
            scale = True
        LAP.loadModel(model)

        best_params_ = LAP.model.best_params_
        for key in best_params_.keys():
            best_params_[key] = [best_params_[key]]
        LAP.n_jobs = n_jobs

        LAP.learnFromVector(
            curX,
            y,
            group=group,
            classifier=self.classifier,
            param_grid=LAP.model.best_params_,
            standardize=scale,
            scoring=self.scoring,
            cv=self.cv)

        modelDir = os.path.join(
            os.path.dirname(
                self.models_path_[0]), 'model_{}.npz'.format(
                self.best_idx_))
        LAP.saveModel(modelDir)
        if save_dir:
            LAP.saveCMFromCV(save_dir)

    def predictRasters(self, inRaster, outRasterPrefix,
                       inMaskRaster=False, confidence=False, modelPath=False):
        """
        Predict each best found features with SFFS.fit(X,y).

        Parameters
        ----------
        inRaster : str.
            Path of the raster to predict.
        outRasterPrefix : str.
            Prefix of each raster to save. Will add in suffix the iteration number then .tif.
            E.g. outRasterPrefix = `classification_`, will give `classification_0.tif` for the first prediction.
        confidence : False or str. Default False.
            If str, same as outRasterPrefix.
        """

        self.__resetMask()

        for idx, model in enumerate(self.models_):
            if self.path_to_save_models  is False:
                LAP = self.models_[idx]
            else:
                LAP = learnAndPredict(n_jobs=1, verbose=self.verbose)
                LAP.loadModel(model)
                
            LAP.customizeX(self.transform, idx=idx, customizeX=True)
                
            outRaster = outRasterPrefix + str(idx) + '.tif'
            
            LAP.predictRaster(inRaster, outRaster,
                              confidence=confidence, inMaskRaster=inMaskRaster)

    def customizeX(self, xFunction, **kwargs):
        self.xFunction = xFunction
        self.xKwargs = kwargs

    def __transformInFit(self, X, idx=0, customizeX=False):
        mask = np.copy(self.mask)
        if customizeX is False:
            fieldsToKeep = self.__convertIdxToNComp(idx)
            mask[fieldsToKeep] = 0
            X = X[:, ~mask]
        if customizeX is True:
            self.mask[self.best_features_[idx]] = 0
            if self.xFunction:
                X = self.xFunction(X, **self.xKwargs)
            X = X[:, ~self.mask]

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        return X

    def transform(self, X, idx=0, customizeX=False):
        """
        Parameters
        ----------
        idx : int, or str with 'best'.
            The idx to return X array
        """
        self.__resetMask()

        self.best_idx_ = np.argmax(self.best_scores_)
        if idx == 'best':
            idx = self.best_idx_
        if self.n_comp > 1:
            for candidate in range(idx + 1):
                if candidate <= idx:
                    idxToMask = [
                        self.best_features_[candidate] *
                        self.n_comp +
                        i for i in range(
                            self.n_comp)]
                    self.mask[idxToMask] = 0
        else:
            self.mask[self.best_features_[:idx + 1]] = 0

        if customizeX is False:
            #            fieldsToKeep = self.__convertIdxToNComp(idx)
            X = X[:, ~self.mask]
#            X = np.hstack((X[:, ~self.mask], xToStack))

        if customizeX is True:
            #            if self.n_comp>1:
            #                idxToMask = [self.best_features_[candidate]*self.n_comp + i for i in range(self.n_comp)]
            #                self.mask[idxToMask] = 0
            #            else:
            #                self.mask[self.best_features_[idx]] = 0
            if self.xFunction:
                X = self.xFunction(X, **self.xKwargs)
            X = X[:, ~self.mask]

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def __getFeatureId(self, candidate):
        """

        """
        return np.where(self.mask == 1)[0][candidate * self.n_comp]

    def __convertIdxToNComp(self, idx):
        """
        """
        idxUnmask = self.__getFeatureId(idx)
        n_features_to_get = [idxUnmask + j for j in range(self.n_comp)]

        return n_features_to_get

    def __maskIdx(self, idx):
        """
        Add the idx to the mask
        """
        self.mask[self.__convertIdxToNComp(idx)] = 0

    def __resetMask(self):
        """
        """
        self.mask[:] = 1

    def getBestModel(self, clone=False):
        self.best_idx_ = np.argmax(self.best_scores_)
        if self.path_to_save_models  :
            LAP = learnAndPredict(n_jobs=1, verbose=self.verbose)
            LAP.loadModel(self.models_path_[self.best_idx_])
        else:
            LAP = self.models_[self.best_idx_]

        return LAP
