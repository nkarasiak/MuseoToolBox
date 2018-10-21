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

import vectorTools
import os
import numpy as np

class samplingMethods:    
    def standCV(inStand,SLOO=True,maxIter=False,seed=None):
        """
        SLOO : float, default True.
            If float from 0.1 to 0.9 (means keep 90% for training)
        maxIter : default False.
            If False : will iterate as many times as the smallest number of stands.
            If int : will iterate the number of stands given in maxIter.
        
        seed : int, default None.
            If seed, int, to repeat exactly the same random.
        """
        samplingType = 'STAND'
        return [samplingType,dict(inStand=inStand,SLOO=SLOO,maxIter=maxIter,seed=seed)]
    
    def SLOOCV(coordinateMatrix,distanceThresold,minTrain=None,SLOO=True,maxIter=False,seed=None):
        """
        coordinateMatrix : array.
            You can generate it via ...
        distanceThresold : int.
            In pixels.
        minTrain : int/float, default None.
            The minimum of training pixel to achieve. if float (0.01 to 0.99) will a percentange of the training pixels.
        SLOO : True or float from 0.1 to 0.9 (means keep 90% for training)
        maxIter : default False.
            If False : will iterate as many times as the smallest number of stands.
            If int : will iterate the number of stands given in maxIter.
        seed : int, default None.
            If seed, int, to repeat exactly the same random.            
        """
        samplingType = 'SLOO'
        return [samplingType,dict(coordinateMatrix=coordinateMatrix,distanceThresold=distanceThresold,minTrain=minTrain,SLOO=SLOO,maxIter=maxIter,seed=seed)]
        
    def random(split=0.5,maxSamples=None,seed=None,nIter=5):
        """
        split : float,int. Default 0.5.
            If float from 0.1 to 0.9 (means keep 90% for training). If int, will try to reach this sample for every class.
        nSamples: int or str. Default None.
            If int, the max samples per class.
            If str, only 'smallest' to sample as the smallest class.
        seed : int, default None.
            If seed, int, to repeat exactly the same random.    
        nIter : int, default 5.
            Number of iteration of the random sampling (will add 1 to the seed at each iteration if defined).
        """
        samplingType = 'random'
        return [samplingType,dict(split=split,maxSamples=maxSamples,seed=seed,nIter=nIter)]

    def extractCoordinatesMatrixForSLOOCV(inVector,inRaster):
        #TODO
        from MuseoToolBox.tools import rasterTools
        return coordinateMatrix



class sampleSelection:
    def __init__(self,samplingMethod):
        """
        sampleSelection generate the duo valid/train samples in order to your samplingMethods choosen function.
        
        Function
        ----------
        getCrossValidation() : Function.
            Get a memory cross validation to use directly in Scikit-Learn.
        
        saveVectorFiles() : Need default output name (str).
            To save as many vector files (train/valid) as your Cross Validation method outputs.
        
        getSupportedExtensions() : Function.
            Show you the list of supported vector extensions type when using saveVectorFiles function.
        
        reinitialize() : Function.
            If you need to regenerate the cross validation, you need to reinitialize it.
            
        """
        self.samplingMethod = samplingMethod
        
        self.extensions = ['sqlite','shp','netcdf','gpx']
        self.driversName = ['SQLITE','ESRI Shapefile','netCDF','GPX']
        
        self.samplingType = samplingMethod[0]
        
        self.__alertMessage = 'It seems you already generated the cross validation. \n Please use reinitialize function if you want to regenerate the cross validation. \n But check if you defined a seed number in your samplingMethods function to have the same output.'
        self.__alreadyRead = False
        
        ### Totally random
        if self.samplingType == 'random':
            FIDs,self.fts,self.srs= vectorTools.readROIFromVector(inVector,inField,getFeatures=True,srs=True)
            FIDs = FIDs.flatten()
            from sklearn.model_selection import train_test_split
            nIter = samplingMethod[1]['nIter']
            seed = samplingMethod[1]['seed']
            train_size = samplingMethod[1]['split']
            maxSamples = samplingMethod[1]['maxSamples']
            if isinstance(maxSamples,str):
                nSamples = np.amin(np.unique(FIDs,return_counts=True)[1])
                def train_test_eq_split(X, y, n_per_class, random_state=None):
                    if random_state:
                        np.random.seed(random_state)
                    sampled = X.groupby(y, sort=False).apply(
                        lambda frame: frame.sample(n_per_class))
                    mask = sampled.index.get_level_values(1)
                
                    X_train = X.drop(mask)
                    X_test = X.loc[mask]
                    y_train = y.drop(mask)
                    y_test = y.loc[mask]

                return X_train, X_test, y_train, y_test
                train_test_eq_split(FIDs,random_state=seed)

            if isinstance(seed,int):
                self.crossvalidation = [train_test_split(FIDs,train_size=train_size,random_state=samplingMethod[1]['seed']+i) for i in range(nIter)]
            else:
                self.crossvalidation = [train_test_split(FIDs,train_size=train_size,random_state=i) for i in range(nIter)]
            
        ### Split at maximum distance beyond each point
        
        ### For Spatial-Leave-One-Out
        if self.samplingType == 'SLOO':
            self.crossvalidation = vectorTools.distanceCV(**samplingMethod[1])
        
        ## For Stand Split    
        if self.samplingType == 'STAND':
            inStand = samplingMethod[1]['inStand']
            SLOO = samplingMethod[1]['SLOO']
            maxIter = samplingMethod[1]['maxIter']
            #FIDs,STDs,srs,fts = vectorTools.readFieldVector(inVector,inField,inStand,getFeatures=True)
            FIDs,STDs,self.fts,self.srs = vectorTools.readROIFromVector(inVector,inField,inStand,getFeatures=True,srs=True)
            FIDs = FIDs.flatten()
            self.crossvalidation = vectorTools.standCV(FIDs,STDs,SLOO=SLOO,maxIter=maxIter)
            
    def reinitialize(self):
        self.__init__(self.outVector,self.samplingMethod)
        
    def getSupportedExtensions(self):
        print('Output extension supported for this class are : ')
        for idx,ext in enumerate(self.extensions):
            print(3*' '+'- '+self.driversName[idx]+' : '+ext)
    
    def getCrossValidation(self):
        if self.__alreadyRead is True:
            print(self.__alertMessage)
        else:
            self.__alreadyRead = True
            return self.crossvalidation
        
    def saveVectorFiles(self,outVector):
        self.__fileName,self.__ext = os.path.splitext(outVector)
        
        if self.__ext[1:] not in self.extensions:
            print('Your extension {} is not recognized as a valid extension for saving shape.'.format(self.__ext))
            self.getSupportedExtensions()
            print('We recommend you to use sqlite extension.')
        
        else:
            if self.__alreadyRead is True:
                print(self.__alertMessage)
            else:
                listOutput = []
                self.cv = []
                for idx,trvl in enumerate(self.crossvalidation):
                    self.cv.append([trvl[0],trvl[1]])
                    for i in trvl[0]:
                        print(i)
                        print(i.shape)
                    trFeat = [self.fts[i] for i in trvl[0]]
                    vlFeat= [self.fts[i] for i in trvl[1]]
                    tr = self.__fileName+'_train_'+str(idx)+self.__ext
                    vl = self.__fileName+'_valid_'+str(idx)+self.__ext
                    self.__saveToShape__(trFeat,self.srs,tr)
                    self.__saveToShape__(vlFeat,self.srs,vl)
                    listOutput.append([tr,vl])
                self.__alreadyRead = True
                return listOutput


    
    def __saveToShape__(self,array,srs,outShapeFile):
        # Parse a delimited text file of volcano data and create a shapefile
        # use a dictionary reader so we can access by field name
        # set up the shapefile driver
        import ogr

        driverIdx=[x for x,i in enumerate(self.extensions) if i==self.__ext[1:]][0]
        outDriver = ogr.GetDriverByName(self.driversName[driverIdx])
        
        # create the data source
        if os.path.exists(outShapeFile):
            outDriver.DeleteDataSource(outShapeFile)
        # Remove output shapefile if it already exists
        
        ds = outDriver.CreateDataSource(outShapeFile) #options = ['SPATIALITE=YES'])
    
        # create the spatial reference, WGS84
        
        lyrout = ds.CreateLayer('randomSubset',srs=srs,geom_type=ogr.wkbPoint)
        fields = [array[1].GetFieldDefnRef(i).GetName() for i in range(array[1].GetFieldCount())]
        if lyrout is None:
            print('fail to save')
        
        
        if self.__ext[1:] != 'shp':
            isShp = False
            lyrout.StartTransaction()
        else:
            isShp = True
            
        for i,f in enumerate(fields):
            field_name = ogr.FieldDefn(f, array[1].GetFieldDefnRef(i).GetType())
            if isShp:
                field_name.SetWidth(array[1].GetFieldDefnRef(i).GetWidth())
            lyrout.CreateField(field_name)
        
        for k in array:
            lyrout.CreateFeature(k)
        k,i,f = None,None,None
        
        if not isShp:
            lyrout.CommitTransaction()
        # Save and close the data source
        ds = None
    
if __name__ == '__main__':
    
    inVector = '/mnt/DATA/Formosat_2006-2014/v2/ROI/ROI_2154.sqlite'
    inField = 'level3'
    inStand = 'spjoin_rif'
    
    
    sampling = samplingMethods.standCV(inStand,SLOO=True,maxIter=3,seed=1)
    sampling = samplingMethods.random('balanced',nIter=10)
    rsv = sampleSelection(sampling)
    files = rsv.saveVectorFiles('/tmp/nana.sqlite')
    """
    cv = rsv.getCrossValidation()
    for tr,vl in cv:
        print(tr.shape)
        print(vl.shape)
    """