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
import numpy as np
class learnAndPredict:
    def __init__(self,classifier,n_jobs=1,param_grid=None):
        """
        Parameters
        ----------
        classifier : str or class from scikit-learn.
            str is only 'GMM'. Else, you can input RandomForestClassifier got from 'from sklearn.ensemble import RandomForestClassifier'
        cv : None, or object.
            if object, from MuseoToolBox.crossValidationSelection.samplingSelection().
        """
            
        self.classifier = classifier
        self.n_jobs = n_jobs
        self.param_grid = param_grid
        
    def learnFromVector(self,X,Y,outModel=None,outValidation=None,outStatsFromCV=True,cv=None):
        self.X = X
        self.Y = Y
        self.__learn__(X,Y,outModel,outStatsFromCV,cv)
        
    def learnFromRaster(self,inRaster,inVector,inField,outModel=None,outValidation=None,outStatsFromCV=True,cv=None):
        from MuseoToolBox.rasterTools import getSamplesFromROI
        X,Y = getSamplesFromROI(inRaster,inVector,inField)
        self.X = X
        self.Y = Y
        self.__learn__(X,Y,outModel,outStatsFromCV,cv)
        
    def __learn__(self,X,Y,outModel,outStatsFromCV,cv):
        self.outStatsFromCV = outStatsFromCV
        from sklearn.model_selection import GridSearchCV
        if cv is None:
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=3)
        
        if outStatsFromCV is True:
            self.CV = []
            if type(cv).__name__ == 'StratifiedKFold':
                cv = cv.split(X,Y)
            for tr,vl in cv: 
                self.CV.append((tr,vl))
        else:
            self.CV = cv
        
        """if len(self.param_grid)==0:
            if self.classifier.__name__ == 'RandomForestClassifier':
                param_grid = dict(n_estimators=3**np.arange(1,5),max_features=range(1,x.shape[1],int(x.shape[1]/3)))                      
            elif self.classifier.__name__ ==  'SVC':
                param_grid = dict(gamma=2.0**np.arange(-4,4), C=10.0**np.arange(-2,5))                 
            elif self.classifier.__name__ == 'KNeighborsClassifier':
                param_grid = dict(n_neighbors = np.arange(1,20,4))            
            else:
                raise Exception('Please define a param_grid')
        """
        if isinstance(self.param_grid,dict):
            grid = GridSearchCV(self.classifier,param_grid=self.param_grid, cv=self.CV,n_jobs=self.n_jobs)
            grid.fit(X,Y)
            self.model = grid.best_estimator_
            self.model.fit(X,Y)
            for key in self.param_grid.keys():
                message = 'best '+key+' : '+str(grid.best_params_[key])
                print(message)

        else:
            self.model = self.classifier.fit(X,Y)
        

    def saveModel(self,path):
        self.model = np.save()
    def loadModel(self,path):
        self.model = np.load()
    def predictFromArray(self,X):
        """
        Predict label from array.
        Parameters
        ----------
        X : array.
            The array to predict. Must have the same number of bands of the initial array/raster.
        """
        Xpredict = self.model.predict(X)
        return Xpredict

    def predictConfidencePerClassFromArray(self,X):
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
        Xpredict = self.model.predict_proba(X)*100        
        return Xpredict

    def predictConfidenceOfHighestClassFromArray(self,X):
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
        Xpredict = np.amax(self.model.predict_proba(X)*100,axis=1)
        return Xpredict

    def predictFromRaster(self,inRaster,outRaster,outConfidencePerClass=False,outConfidence=False,inMaskRaster=False,outNoData=0):
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
        
        from MuseoToolBox.rasterTools import rasterMath
        from MuseoToolBox.rasterTools import getGdalDTFromMinMaxValues
        rM = rasterMath(inRaster,inMaskRaster,'Prediction... ')
        gdalDT = getGdalDTFromMinMaxValues(int(np.amax(np.unique(self.Y))))
        rM.addFunction(self.predictFromArray,outRaster,1,gdalDT,outNoData=0)
        if outConfidence:
            rM.addFunction(self.predictConfidenceOfHighestClassFromArray,outConfidence,outNBand=False,outGdalDT=3,outNoData=-9999)    
        if outConfidencePerClass:
            rM.addFunction(self.predictConfidencePerClassFromArray,outConfidencePerClass,outNBand=False,outGdalDT=3,outNoData=-9999)    
        rM.run()
        
    def getStatsFromCV(self,confusionMatrix=True,kappa=False,OA=False,F1=False):
        if self.outStatsFromCV is False:
            raise Exception('outStatsFromCV in fromRaster or fromVector must be True')
        else:
            from MuseoToolBox.stats.statsFromConfusionMatrix import confusionMatrix#,statsFromConfusionMatrix,
            CM = [[]]
            kappas=[]
            OAs=[]
            F1s=[]
            for train_index, test_index in self.CV:
                X_train, X_test = self.X[train_index], self.X[test_index]
                Y_train, Y_test = self.Y[train_index], self.Y[test_index]
                    
                self.model.fit(X_train, Y_train)
                X_pred = self.model.predict(X_test)
                cmObject = confusionMatrix(Y_test,X_pred,kappa=kappa,OA=OA,F1=F1)
                cm = cmObject.confusion_matrix
                CM[0].append([cm])
                if kappa:
                    kappas.append(cmObject.Kappa) #statsFromConfusionMatrix(cm).__get_kappa()
                if OA :
                    OAs.append(cmObject.OA) #statsFromConfusionMatrix(cm).__get_OA()
                if F1 : 
                    F1s.append(cmObject.F1) #statsFromConfusionMatrix(cm).__get_F1()
            
            toReturn = CM
            if kappa is True:
                toReturn.append(kappas)
            if OA is True:
                toReturn.append(OAs)
            if F1 is True:
                toReturn.append(F1s)
            return toReturn
            