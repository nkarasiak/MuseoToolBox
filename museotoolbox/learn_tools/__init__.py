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
from __future__ import absolute_import, print_function, division
from joblib import Parallel, delayed
import os
import numpy as np
from sklearn import metrics
from sklearn.base import clone,BaseEstimator
from sklearn import warnings
from ..internal_tools import progressBar

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
        >>> RS50 = mtb.cross_validation.RandomCV(valid_size=0.5,n_splits=10,
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
            cv=False,
            scoring='accuracy', **gridSearchCVParams):
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
            cv,
            scoring,
            **gridSearchCVParams)

    def learnFromRaster(
            self,
            inRaster,
            inVector,
            inField,
            classifier,
            inGroup=False,
            param_grid=None,
            scale=False,
            cv=False,
            scoring='accuracy',
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
            cv,
            scoring,
            **gridSearchCVParams)

    def __learn__(self, X, y, groups, classifier,
                  param_grid, cv, scoring='accuracy', **gridSearchCVParams):

        if cv is not False and not isinstance(cv, int):
            self.CV = []
            for tr, vl in (cv for cv in cv.split(
                    X, y, groups) if cv is not None):
                self.CV.append((tr, vl))
        else:
            self.CV = cv

        from sklearn.model_selection import GridSearchCV

        if isinstance(param_grid, dict):
            self.model = GridSearchCV(
                self.classifier,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
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
                self.scale = True
                self.model, self.scaler = np.load(path)
            else:
                self.scale = False

        else:
            if not path.endswith('npz'):
                path += '.npz'
            model = np.load(path)
            self.__dict__.update(model['LAP'].tolist())

    def __convertX(self, X, **kwargs):
        if 'Xfunction' in kwargs:
            Xfunction = kwargs['Xfunction']
            kwargs.pop('Xfunction', None)
            X = Xfunction(X, **kwargs)

        if self.scale:
            X = self.scaler.transform(X)
        return X

    def predictArray(self, X, **kwargs):
        """
        Predict label from array.

        Parameters
        ----------
        X : array.
            The array to predict. Must have the same number of bands of the initial array/raster.
        **kwargs :
            Xfunction : a custom function to modify directly the array from the raster.
        """

        X = self.__convertX(X, **kwargs)

        self.Xpredict = self.model.predict(X)
        return self.Xpredict

    def predictConfidencePerClass(self, X, **kwargs):
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
        self.__convertX(X, **kwargs)

        Xpredict_proba = self.model.predict_proba(X) * 100
        if Xpredict_proba.ndim == 1:
            Xpredict_proba = Xpredict_proba.reshape(-1, 1)
        # share prediction in class in case of confidence for only predicted
        # class
        self.Xpredict_proba = Xpredict_proba
        return Xpredict_proba

    def predictConfidenceOfPredictedClass(self, X, **kwargs):
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
            self.__convertX(X, **kwargs)
            Xpredict_proba = np.amax(self.model.predict_proba(X) * 100, axis=1)
        return Xpredict_proba

    def predictRaster(
            self,
            inRaster,
            outRaster,
            confidencePerClass=False,
            confidence=False,
            inMaskRaster=False,
            outNoData=0,
            **kwargs):
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
            Get numpy datatype throught : convertGdalAndNumpyDataType(getGdalDTFromMinMaxValues(maximumClassValue)))
        outNoData : int, default 0.
            Value of no data for the outRaster.
        """

        from ..raster_tools import rasterMath, getGdalDTFromMinMaxValues, convertGdalAndNumpyDataType
        rM = rasterMath(inRaster, inMaskRaster, message='Prediction... ')

        numpyDT = convertGdalAndNumpyDataType(
            getGdalDTFromMinMaxValues(np.amax(self.model.classes_)))

        rM.addFunction(
            self.predictArray,
            outRaster,
            outNBand=1,
            outNumpyDT=numpyDT,
            outNoData=outNoData,
            **kwargs)

        if confidencePerClass:
            rM.addFunction(
                self.predictConfidencePerClass,
                confidencePerClass,
                outNBand=False,
                outNumpyDT=np.int16,
                outNoData=np.iinfo(np.int16).min,
                **kwargs)

        if confidence:
            rM.addFunction(
                self.predictConfidenceOfPredictedClass,
                confidence,
                outNBand=1,
                outNumpyDT=np.int16,
                outNoData=np.iinfo(np.int16).min,
                **kwargs)
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

class sequentialFeatureSelection(BaseEstimator):
    """
    Sequential Feature Selection 
    
    Parameters
    ----------
    classifier : array.
        The feature array, each column represent an idx
    param_gridy : array.
    cv : int, default 1.
        The number of component per feature. If 4, each feature has 4 columns.
    """
    def __init__(self,classifier,param_grid,cv=5,scoring='accuracy',n_comp=1,verbose=1):
        # share args
        self.n_comp = n_comp
        self.classifier = classifier
        self.param_grid = param_grid
        self.scoring=scoring
        self.cv = cv
        self.verbose = verbose
        
        self.xFunction = False 
        self.xKwargs = False
        
    def fit(self,X,y,group=False,pathToSaveModels=False,n_jobs=1):
        """
        Parameters
        ----------
        X : 
        y : 
        g: group
            
        """
        self.X = X
        self.X_ = np.copy(X)
        self.y = y
        self.group = group
        
        self.models_path_ = []
        
        if self.xFunction:
            self.X = self.xFunction(X,**self.xKwargs)            

        xSize = self.X.shape[1]
        self.n_features = int(xSize/self.n_comp)
        totalIter = np.sum([self.n_features-i for i in range(self.n_features)])
        if self.verbose:
            pB = progressBar(totalIter,message='SFFS:')
        self.mask = np.ones(xSize,dtype=bool)

        self.models_,self.best_scores_,self.best_features_ = [[],[],[]]
        self.subsets_ = dict()
        
        for j in range(self.n_features):    
            resPerFeatures = list()
            #bestScore,LAPs,bestParams,cvResults,models = [[],[],[],[],[]dd]
            n_features_to_test = int(self.X[:,self.mask].shape[1]/self.n_comp)
            if pathToSaveModels:
                all_scores_file = pathToSaveModels+'all_scores_{}.csv'.format(j)
                if os.path.exists(all_scores_file):
                    print('Feature {} already computed'.format(j))
                    scores = np.loadtxt(all_scores_file,delimiter=',')
                    
                    if scores.ndim==1:
                        all_scores = [scores[1]]
                        best_candidate = 0
                    else:
                        all_scores = scores[:,1]
                        best_candidate = np.argmax(scores[:,1])
                    LAP = learnAndPredict(n_jobs=n_jobs,verbose=self.verbose-1)
                    LAP.loadModel(pathToSaveModels+'model_{}.npz'.format(j))
                    self.models_path_.append(pathToSaveModels+'model_{}.npz'.format(j))

            for idx in range(n_features_to_test): # at each loop, remove best candidate                    
                if self.verbose:
                    pB.addPosition()
                LAP = learnAndPredict(n_jobs=n_jobs,verbose=self.verbose-1)        
                curX = self.__getX(self.X,idx)
                if self.xFunction:
                    scale=False
                else:
                    scale=True
                LAP.learnFromVector(curX,y,group=group,classifier=self.classifier,param_grid=self.param_grid,scale=scale,scoring=self.scoring,cv=self.cv)
                
                resPerFeatures.append(LAP)
                
            all_scores = [np.amax(LAP.model.best_score_) for LAP in resPerFeatures]
            best_candidate = np.argmax(all_scores)
            # self.__bestLAPs.append(resPerFeatures[best_candidate])
            LAP = resPerFeatures[best_candidate]

            if pathToSaveModels:
                if not os.path.exists(os.path.join(pathToSaveModels,str(j))):
                    os.makedirs(os.path.join(pathToSaveModels,str(j)))
                LAP.saveModel(pathToSaveModels+'model_{}.npz'.format(j))
                LAP.saveCMFromCV(os.path.join(pathToSaveModels,str(j)),n_jobs=-1)
                scoreWithIdx = np.hstack((np.where(self.mask==1)[0].reshape(-1,1),np.asarray(all_scores,dtype=np.float32).reshape(-1,1)))
                np.savetxt(all_scores_file,scoreWithIdx,fmt='%0.d,%.4f')
            
            self.models_.append(resPerFeatures[best_candidate].model)
                
                # store results
            best_feature_id = int(self.__getFeatureId(best_candidate)/self.n_comp)
            self.best_scores_.append(all_scores[best_candidate])
            self.best_features_.append(best_feature_id)
                
            if self.verbose:
                print('\nBest feature with %s feature(s) : %s' %(j+1,best_feature_id))
                print('Best mean score : %s' %np.amax(all_scores))
    
            self.subsets_[str(j)] = dict(avg_score=np.amax(all_scores),
                         feature_idx=self.best_features_.copy(),
                         cv_score=LAP.model.cv_results_,
                         best_score_=np.amax(all_scores),
                         best_feature_=best_feature_id)
                
            self.__maskIdx(best_candidate)
    
    def predictRasters(self,inRaster,outRasterPrefix,inMaskRaster=False,confidence=False,modelPath=False):
        """
        Predict each best found features with SFFS.fit(X,y).
        
        Parameters
        ----------
        inRaster : str.
            Path of the raster to predict.
        outRasterPrefix : str.
            Prefix of each raster to save. Will add in suffix the iteration number then .tif.
            E.g. outRasterPrefix = 'classification_', will give 'classification_0.tif' for the first prediction.
        confidence : False or str. Default False.
            If str, same as outRasterPrefix.
        """        
        self.__resetMask()
        
        if len(self.models_path_) == 0:
            print('Warning : You have to define a path to save model in the `fit` function in order to predict rasters.')
        
        for idx,model in enumerate(self.models_path_):
            print(self.mask)
            LAP = learnAndPredict(n_jobs=1,verbose=self.verbose)
            LAP.loadModel(model)
            
            outRaster = outRasterPrefix+str(idx)+'.tif'
            LAP.predictRaster(inRaster,outRaster,confidence=confidence,inMaskRaster=inMaskRaster,
                              Xfunction=self.__getX,idx=idx,customizeX=True)
                
    def customizeX(self,xFunction,**kwargs):
        self.xFunction = xFunction
        self.xKwargs = kwargs
        
    def __getX(self,X,idx=0,customizeX=False):
        """
        Parameters
        ----------
        idx : int.
            The idx to return X array
        """
        if customizeX is False:
            fieldsToKeep = self.__convertIdxToNComp(idx)
            xToStack = X[:,fieldsToKeep]
            if xToStack.ndim == 1: 
                xToStack = xToStack.reshape(-1,1)        
            X = np.hstack((X[:,~self.mask],xToStack))
        
        if customizeX is True:
            self.mask[self.best_features_[idx]] = 0
            if self.xFunction:
                X = self.xFunction(X,**self.xKwargs)
            X = X[:,~self.mask]

        if X.ndim==1:
            X = X.reshape(-1,1)
        
        return X
        
    def __getFeatureId(self,best_candidate):
        """
        
        """
        return np.where(self.mask==1)[0][best_candidate*self.n_comp]
     
    def __convertIdxToNComp(self,idx):
        """
        """
        idxUnmask = self.__getFeatureId(idx)
        n_features_to_get = [idxUnmask+j for j in range(self.n_comp)]
        return n_features_to_get
    
    def __maskIdx(self,idx):
        """
        Add the idx to the mask
        """
        self.mask[self.__convertIdxToNComp(idx)] = 0
    
    def __resetMask(self):
        """
        """
        self.mask[:] = 1
