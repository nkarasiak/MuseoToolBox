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

from __future__ import absolute_import
from MuseoToolBox.tools import vectorTools,rasterTools
import os

### TODO
### Rasterize vector to keep only the centroid !


class samplingMethods:    
    def standCV(inStand,SLOO=True,maxIter=False,seed=None):
        """
        Generate a Cross-Validation by stand/polygon group.

        Parameters
        ----------
        inStand : str.
            field name containing label of group/stand.
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
    
    def farthestCV(distanceMatrix,):
        """
        Generate a Cross-Validation using the farthest distance between the training and validation samples.

        Parameters
        ----------
        """        
        samplingType = 'farthestCV'
        return [samplingType,dict()]
    def SLOOCV(inRaster,inVector,distanceThresold,minTrain=None,SLOO=True,maxIter=False,seed=None):
        """
        Generate a Cross-Validation with Spatial Leave-One-Out method.
        
        See : https://doi.org/10.1111/geb.12161.

        Parameters
        ----------
        inRaster : str.
            Path of the raster to compute the distance in pixel.
        inVector : str.
            Path of the vector files to compute distance between each feautre.
        distanceThresold : int.
            In pixels.
        minTrain : int/float, default None.
            The minimum of training pixel to achieve. if float (0.01 to 0.99) will a percentange of the training pixels.
        SLOO : True or float
            from 0.0 to 1.0 (means keep 90% for training). If True, keep only one sample per class for validation.
        maxIter : default False.
            If False : will iterate as many times as the smallest number of stands.
            If int : will iterate the number of stands given in maxIter.
        seed : int, default None.
            If seed, int, to repeat exactly the same random.            
        """
        samplingType = 'SLOO'
        distanceMatrix = samplingMethods.getDistanceMatrixForDistanceCV(inVector,inRaster)
        return [samplingType,dict(inRaster=inRaster,inVector=inVector,distanceMatrix=distanceMatrix,distanceThresold=distanceThresold,minTrain=minTrain,SLOO=SLOO,maxIter=maxIter,seed=seed)]
        
    def randomCV(train_size=0.5,seed=None,nIter=5):
        """
        split : float,int. Default 0.5.
            If float from 0.1 to 0.9 (means keep 90% per class for training). If int, will try to reach this sample for every class.
        nSamples: int or str. Default None.
            If int, the max samples per class.
            If str, only 'smallest' to sample as the smallest class.
        seed : int, default None.
            If seed, int, to repeat exactly the same random.    
        nIter : int, default 5.
            Number of iteration of the random sampling (will add 1 to the seed at each iteration if defined).
        """
        samplingType = 'random'
        return [samplingType,dict(train_size=train_size,seed=seed,nIter=nIter)]

    def getDistanceMatrixForDistanceCV(inVector,inRaster):
        #TODO
        import tempfile
        tempTif = tempfile.mktemp('_roi.tif')
        tempTif = rasterTools.rasterize(inRaster,inVector,None,tempTif)
        coords = rasterTools.get_samples_from_roi(inRaster,tempTif,getCoords=True,onlyCoords=True)
        os.remove(tempTif)
        
        distanceMatrix = vectorTools.distMatrix(coords)

        return distanceMatrix


class sampleSelection(samplingMethods):
    def __init__(self,inVector,inField,samplingMethod):
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
        
        # create unique randil state if no seed
        if self.samplingMethod[1]['seed'] is None:
            import time
            self.samplingMethod[1]['seed'] = int(time.time())
            
        ### Totally random
        if self.samplingType == 'random':
            FIDs,self.fts,self.srs= vectorTools.readValuesFromVector(inVector,inField,getFeatures=True)
            FIDs = FIDs.flatten()
            
            nIter = samplingMethod[1]['nIter']
            seed = samplingMethod[1]['seed']
            train_size = samplingMethod[1]['train_size']
            
            self.crossvalidation = [vectorTools.randomPerClass(FIDs,train_size,seed+i) for i in range(nIter)]
        
        ### Split at maximum distance beyond each point
        ### For Spatial-Leave-One-Out
        if self.samplingType == 'SLOO':
            import tempfile
            tempTif = tempfile.mktemp('_roi.tif')
            tempTif = rasterTools.rasterize(inRaster,inVector,inField,tempTif)
            X,Y = rasterTools.get_samples_from_roi(inRaster,tempTif)
            os.remove(tempTif)
            dictForSLOO = {}
            for key,value in samplingMethod[1].items():
                if not key is 'inRaster' and not key is 'inVector' :
                    dictForSLOO[key] = value
            self.crossvalidation = vectorTools.distanceCV(Y=Y,**dictForSLOO)
            
        ## For Stand Split    
        if self.samplingType == 'STAND':
            inStand = samplingMethod[1]['inStand']
            SLOO = samplingMethod[1]['SLOO']
            maxIter = samplingMethod[1]['maxIter']
            #FIDs,STDs,srs,fts = vectorTools.readFieldVector(inVector,inField,inStand,getFeatures=True)
            FIDs,STDs,self.fts,self.srs = vectorTools.readValuesFromVector(inVector,inField,inStand,getFeatures=True)
            FIDs = FIDs.flatten()
            self.crossvalidation = vectorTools.standCV(FIDs,STDs,SLOO=SLOO,maxIter=maxIter)
            
    def reinitialize(self):
        self.__init__(self.outVector,self.samplingMethod)
        
    def getSupportedExtensions(self):
        print('Output extension supported for this class are : ')
        for idx,ext in enumerate(self.extensions):
            print(3*' '+'- '+self.driversName[idx]+' : '+ext)
    
    def getCrossValidationForScikitLearn(self):
        if self.__alreadyRead is True:
            raise Warning(self.__alertMessage)
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
                    trFeat = [self.fts[int(i)] for i in trvl[0]]
                    vlFeat= [self.fts[int(i)] for i in trvl[1]]
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
    
    inRaster = '/mnt/DATA/Formosat_2006-2014/v2/SITS_2014.tif'
    #rasterTools.getCentroidValue(inVector,inRaster)
    
    distanceMatrix = samplingMethods.getDistanceMatrixForDistanceCV(inVector,inRaster)
    sampling = samplingMethods.randomCV(50,5)
    #sampling = samplingMethods.SLOOCV(inRaster,inVector,distanceThresold=450,maxIter=5)
    #sampling = samplingMethods.standCV(inStand,SLOO=True,maxIter=3,seed=1)
    #sampling = samplingMethods.randomCV(maxTrainSamplesPerClass=10)
    rsv = sampleSelection(inVector,inField,sampling)
    #cv = rsv.getCrossValidationForScikitLearn()
    files = rsv.saveVectorFiles('/tmp/test.sqlite')
