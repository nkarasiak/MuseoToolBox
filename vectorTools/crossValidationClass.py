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

class distanceCV:
    def __init__(self,distanceMatrix,Y,distanceThresold=1000,minTrain=-1,SLOO=True,maxIter=False,furtherSplit=False,onlyVaryingTrain=False,stats=False,verbose=False,seed=False,otherLevel=False):
        """Compute train/validation array with Spatial distance analysis.
        
        Object stops when less effective class number is reached (45 loops if your least class contains 45 ROI).
        
        Parameters
        ----------
        distanceMatrix : array
            Matrix distance
        
        Y : array-like
            contain classe for each ROI. Same effective as distanceArray.
            
        distanceThresold : int or float
            Distance(same unit of your distanceArray)
        
        minTrain : int or float
            if >1 : keep n ROI beyond distanceThresold
            if float <1 : minimum percent of samples to use for traning. Use 1 for use only distance Thresold.
            if -1 : same size
        SLOO : Spatial Leave One Out, keep on single validation pixel.
            SLOO=True: Skcv (if maxIter=False, skcv is SLOO from Kevin Le Rest, or SKCV from Pohjankukka)
        maxIter :
            False : as loop as min effective class
        
        
        Returns
        -------
        train : array 
            List of Y selected ROI for train
        
        validation : array 
            List of Y selected ROI for validation
            
        """
        self.distanceArray = distanceMatrix
        self.distanceThresold = distanceThresold
        self.label = np.copy(Y)
        self.T = np.copy(Y)
        self.iterPos = 0
        self.minTrain = minTrain
        if self.minTrain is None:
            self.minTrain = -1
        self.onlyVaryingTrain = onlyVaryingTrain
        if self.onlyVaryingTrain :
            self.validation = np.array([]).astype('int')
        self.minEffectiveClass = min([len(Y[Y==i]) for i in np.unique(Y) ])
        if maxIter :
            self.maxIter = maxIter
        else :
            self.maxIter = self.minEffectiveClass
        self.otherLevel = otherLevel
        self.SLOO = SLOO #Leave One OUT
        self.verbose = verbose
        self.furtherSplit = furtherSplit
        self.stats = stats
        
        if seed:
            np.random.seed(seed)
        else:
            import time
            seed= int(time.time())
        self.seed = seed
        
    def __iter__(self):
        return self
        
    # python 3 compatibility
    def __next__(self):        
        return self.next()
        
    def next(self):
        
        #global CTtoRemove,trained,validate,validation,train,CT,distanceROI
       
        if self.iterPos < self.maxIter:
            np.random.seed(self.seed)
            self.seed += 1
            ROItoRemove = []
            for iterPosition in range(self.maxIter):      
                if self.verbose:print((53*'=' + '\n')*4)
                
                validation = np.array([]).astype('int')
                train = np.array([]).astype('int')
                
                #sp.random.shuffle(self.T)
                
                if self.stats:
                        #Cstats = sp.array([]).reshape(0,9   )
                        Cstats = []
                        
                for C in np.unique(self.label):
                    
                    CT = np.where(self.T==C)[0] # Y is True, where C is the unique class
                    
                    CTtemp = np.copy(CT)   
                    
                    if self.verbose:
                            print('C = '+str(C))
                            print('len total class : '+str(len(CT)))
                            fmt = '' if self.minTrain > 1 else '.0%'
                            
                    trained = np.array([]).astype('int')
                    validate = np.array([]).astype('int')
                    
              
                    while len(CTtemp) > 0:
                    #totalC = len(self.label[self.label==int(C)])
                    #uniqueTrain = 0
                        np.random.seed(self.seed)
                        self.ROI = np.random.permutation(CTtemp)[0] # randomize ROI choice

                        
                        #while uniqueTrain <(self.split*totalC) :
                        #sameClass = sp.where( self.Y[CT] == C )
                        distanceROI = (self.distanceArray[int(self.ROI),:])[CTtemp] # get line of distance for specific ROI
                        
                        if self.minTrain == -1 :
                            #distToCutValid = sp.sort(distanceROI)[:self.maxIter][-1] # get distance where to split train/valid
                            #distToCutTrain = sp.sort(distanceROI)[-self.maxIter:][0] # get distance where to split train/valid
                            
                            trainedTemp = np.array([self.ROI])
                                                     
                            validateTemp = CTtemp[CTtemp!=trainedTemp][distanceROI[distanceROI>0.1]>=self.distanceThresold]
                            # CTtemp[distanceROI>=self.distanceThresold] # trained > distance to cut 
                        
                        if self.maxIter == self.minEffectiveClass:
                            trainedTemp = np.array([self.ROI])

                            validateTemp = CTtemp[distanceROI>self.distanceThresold]
                            #trainedTemp = trainedTemp[trainedTemp!=self.ROI]
                            
                        """
                        elif self.SLOO:                            
                            validateTemp = sp.array([self.ROI]) # validate ROI                            
                            trainedTemp = CTtemp[(distanceROI>=self.distanceThresold)] # Train in a buffer
                        """
#trainedTemp = sp.array([self.ROI]) # train is the current ROI                         
                        """
                        if self.SLOO is True and self.maxIter != self.minEffectiveClass:
                            
                            CTtoRemove = np.concatenate((validateTemp,trainedTemp))                        
                    
                            # Remove ROI for further selection ROI (but keep in Y list)                        
                            for i in np.nditer(CTtoRemove):
                                CTtemp = np.delete(CTtemp,np.where(CTtemp==i)[0])
                        
                            #if self.verbose : print('len CTtemp is : '+str(len(CTtemp)))
                                
                            trained = np.concatenate((trained,trainedTemp))
                            validate = np.concatenate((validate,validateTemp))
                            

                        else:
                            trained = trainedTemp
                            validate = validateTemp
                            

                            
                            CTtemp = []
                        """
                        if self.furtherSplit:
                            validateTemp = np.array([self.ROI])
                            trainedTemp = CTtemp[CTtemp!=validateTemp]

                        trained = trainedTemp
                        validate = validateTemp
                                
                        CTtemp = []
                        #print len(validate)
                    initTrain = len(trained)
                    initValid = len(validate)

                    if self.minTrain > 1:
                        if len(trained) != self.minTrain:
                            
                            indToCut = len(CT)-int(self.minTrain) # get number of ROI to keep
                            distToCut = np.sort(distanceROI)[indToCut] # get distance where to split train/valid
                            trained = CT[distanceROI>=distToCut] # trained > distance to cut 
                            
                            if self.SLOO:  # with SLOO we keep 1 single validation ROI
                                trained = np.random.permutation(trained)[0:self.minTrain]
                            else:
                                if self.verbose:
                                    print('len validate before split ('+ format(self.minTrain,fmt)+') : '+str(len(validate)))
                                validate = CT[distanceROI<=distToCut]
                                
                    elif self.onlyVaryingTrain:
                        if self.verbose:
                            print('only varying train size : First Time init')
                        if len(validate) > int(self.onlyVaryingTrain*len(CT)) :
                            nValidateToRemove = int(len(validate) - self.onlyVaryingTrain*len(CT))
                            indToMove = np.random.permutation(trained)[:nValidateToRemove]
                            for i in indToMove:
                                validate = np.delete(trained,np.where(trained==i)[0])
                            trained = np.concatenate((trained,indToMove))
                            
                        elif len(validate) < int(self.onlyVaryingTrain*len(CT)):
                            nValidToAdd = int(self.minTrain*len(CT) - len(trained))
                            
                            indToMove = np.random.permutation(validate)[:nValidToAdd]
                            for i in indToMove:
                                trained = np.delete(validate,np.where(validate==i)[0])
                            validate = np.concatenate((trained,indToMove))
                        
                        elif len(trained) > int(self.minTrain*(len(CT))):
                            nTrainToRemove = int(self.minTrain*len(CT) - len(trained))
                            
                            indToMove = np.random.permutation(validate)[:nTrainToRemove]
                            for i in indToMove:
                                trained = np.delete(validate,np.where(validate==i)[0])
                            validate = np.concatenate((trained,indToMove))
                            
                        elif len(trained) < int(self.minTrain*(len(CT))):
                            nTrainToAdd = int(self.minTrain*len(CT) - len(trained))
                            
                            indToMove = np.random.permutation(validate)[:nTrainToAdd]
                            for i in indToMove:
                                validate = np.delete(validate,np.where(validate==i)[0])
                            trained = np.concatenate((trained,indToMove))
                        
                    
                    elif self.minTrain != -1 and self.minTrain != 0 and not self.onlyVaryingTrain :
                        initTrain = len(trained)
                        initValid = len(validate)
                        if (len(trained) != self.minTrain*len(CT)) or (self.SLOO and len(trained)==0): # if train size if less than split% of whole class (i.e. 30% for exemple)
                            
                            if self.verbose:
                                print('len trained before '+ format(self.minTrain,fmt)+' : '+str(len(trained)))
                            
                            #distanceROI = (self.distanceArray[int(self.ROI),:])[CT]
                            if self.furtherSplit :
                                if len(trained) > self.minTrain*len(CT):
                                    nTrainToRemove = int(len(trained) - self.minTrain*len(CT))
                                    distanceROI = (self.distanceArray[int(np.random.permutation(trained)[0]),:])[trained]
                                    
                                    distToMove = np.sort(distanceROI)[nTrainToRemove]
                                    #indToMove = distToMove[distanceROI]
                                    indToMove = trained[distanceROI>distToMove] # trained > distance to cut 
                                    for i in indToMove:
                                        trained = np.delete(trained,np.where(trained==i)[0])
                                    validate = np.concatenate((validate ,indToMove))
                                else:
                                    nTrainToAdd = int(self.minTrain*len(CT) - len(trained))
                                    distanceROI = (self.distanceArray[int(np.random.permutation(validate)[0]),:])[validate]
                                    distToMove = np.sort(distanceROI)[-nTrainToAdd]
                                    #indToMove = distToMove[distanceROI]
                                    indToMove = validate[distanceROI>=distToMove] # trained > distance to cut 
                                    for i in indToMove:
                                        validate = np.delete(validate,np.where(validate==i)[0])
                                    trained = np.concatenate((trained,indToMove))
                            
                            else:
                                if len(trained) > self.minTrain*len(CT) :
                                    nTrainToRemove = int(len(trained) - self.minTrain*len(CT))
                                    indToMove = np.random.permutation(trained)[:nTrainToRemove]
                                    for i in indToMove:
                                        trained = np.delete(trained,np.where(trained==i)[0])
                                    validate = np.concatenate((validate ,indToMove))
                                
                                else:
                                    nTrainToAdd = int(self.minTrain*len(CT) - len(trained))
                                    
                                    indToMove = np.random.permutation(validate)[:nTrainToAdd]
                                    for i in indToMove:
                                        validate = np.delete(validate,np.where(validate==i)[0])
                                    trained = np.concatenate((trained,indToMove))
                                
                    if self.stats:                    
                        
                        #CTtemp = sp.where(self.label[trained]==C)[0]
                        CTdistTrain=np.array(self.distanceArray[trained])[:,trained]
                        if len(CTdistTrain) > 1:
                            CTdistTrain=np.mean(np.triu(CTdistTrain)[np.triu(CTdistTrain)!=0])
                        
                        #CTtemp = sp.where(self.label[validate]==C)[0]
                        CTdistValid=np.array(self.distanceArray[validate])[:,validate]
                        CTdistValid=np.mean(np.triu(CTdistValid)[np.triu(CTdistValid)!=0])
                        Cstats.append([self.distanceThresold,self.minTrain,C,initValid,initTrain,len(trained)-initTrain,CTdistTrain,CTdistValid])


                 
                    if self.verbose:
                        print('len validate : '+str(len(validate)))
                        print('len trained : '+str(len(trained)))
                        
                        
                    
                    validation = np.concatenate((validation,validate))                    
                    train = np.concatenate((train,trained))
                    #allDist[sp.where(y[allDist]==C)[0]]
                    #T = sp.searchsorted(T,currentClass)
                    #for i in sp.nditer(train):
                    
                    ROItoRemove.append(validation) # remove current validation ROI
                    ROItoRemove.append(train)
                
 
                    #Cstats = sp.vstack((Cstats,(self.distanceThresold,self.minTrain*100,C,initTrain,initValid,len(trained)-initTrain,len(validate)-initValid,meanDistTrain,meanDistValidation)))
                    
                if self.stats is True:
                    np.savetxt(self.stats,Cstats,fmt='%d',delimiter=',',header="Distance,Percent Train, Label,Init train,Init valid,Ntrain Add,Mean DisT Train,Mean Dist Valid")
        
                    #if not self.SLOO:
                        #validate = CT[distanceROI<distToCut]
                    
                    
                self.iterPos += 1  
                
                if self.verbose:
                    print(53*'=')
                # Remove ROI for further selection ROI (but keep in Y list)
                """
                for i in ROItoRemove:
                    self.T = sp.delete(self.T,sp.where(self.T==i)[0])
                """
                if self.stats and self.stats is True :
                    return validation,train,Cstats
                else:                    
                    return validation,train
            
        else:
            raise StopIteration()
            
            
def distMatrix(inCoords,distanceMetric=False):
    """
    Compute distance matrix between points
    coords : nparray shape[nPoints,2], with first column X, and Y. Proj 4326(WGS84)
    Return matrix of distance matrix between points.
    """
    if distanceMetric:
        from pyproj import Geod
        geod = Geod(ellps='WGS84')
    
        distArray = np.zeros((len(inCoords),len(inCoords)))
        for n,p in enumerate(np.nditer(inCoords.T.copy(), flags=['external_loop'], order='F')):
            for i in range(len(inCoords)):
                x1,y1 = p
                x2,y2 = inCoords[i]
                angle1,angle2,dist = geod.inv(x1,y1,x2,y2)
            
                distArray[n,i] = dist
    
    else:
        from scipy.spatial import distance
        
        distArray = np.array(distance.cdist(inCoords,inCoords,'euclidean'),dtype=int)

    return distArray


class randomPerClass:
    """
    Random array according to FIDs.
    
    Parameters
    ---------
    FIDs : arr.
        Label for each feature.
    train_size : float (<1.0) or int (>1).
        Percentage to keep for training or integer.
    seed : int.
        random_state for numpy.
    """
    def __init__(self,FIDs,train_size=0.5,nIter=5,seed=None):
        self.FIDs = FIDs
        self.train_size = train_size
        self.nIter = nIter
        if seed:
            np.random.seed(seed)
        else:
            import time
            seed= int(time.time())
        self.seed = seed
        
        self.iter = 0
    def __iter__(self):
        return self
        
    # python 3 compatibility
    def __next__(self):        
        return self.next()
        
    def next(self):
        if self.iter < self.nIter:
            train,valid = [np.asarray([]),np.asarray([])]
            for C in np.unique(self.FIDs):
                Cpos=np.where(self.FIDs==C)[0]
                
                if self.train_size < 1:
                    toSplit = int(self.train_size*len(Cpos))
                else:
                    toSplit = self.train_size
                    
                np.random.seed(self.seed)
                tempTrain = np.random.permutation(Cpos)[:toSplit]
                TF = np.in1d(Cpos,tempTrain,invert=True)
                tempValid = Cpos[TF]
                train = np.concatenate((train,tempTrain))
                valid = np.concatenate((valid,tempValid))
            
            
            self.seed += 1
            self.iter += 1
            
            return train,valid
        else:
             raise StopIteration()
             



class standCV:
    def __init__(self,Y,stand,maxIter=False,SLOO=True,seed=False):
        """Compute train/validation per stand.
        Y : array-like
            contains class for each ROI.
        Stand : array-like
            contains stand number for each ROI.
        maxIter : False or int
            if False, maxIter is the minimum stand number of all species.
        SLOO :  Bool
            True  or False. If SLOO, keep only one Y per validation stand.
        """
        self.Y = Y
        self.uniqueY = np.unique(self.Y)
        self.stand = stand
        self.SLOO = SLOO
        
        if type(SLOO) == bool:
            self.split = 0.5
            
        else:
            self.split = self.SLOO
        self.maxIter=maxIter
        self.iterPos = 0
        
        if seed:
            np.random.seed(seed)
        else:
            import time
            seed= int(time.time())
        self.seed = seed
        if maxIter :
            self.maxIter = maxIter
        else:

            maxIter = []
            for i in np.unique(Y):
                standNumber = np.unique(np.array(stand)[np.where(np.array(Y)==i)])
                maxIter.append(standNumber.shape[0])
            self.maxIter = np.amin(maxIter)
        
            
    def __iter__(self):
        return self
        
    # python 3 compatibility
    def __next__(self):
        return self.next()
    
    def next(self):
         self.seedidx = 0
         if self.iterPos < self.maxIter:
             StandToRemove = []
             train = np.array([],dtype=int)
             validation = np.array([],dtype=int)
             for i in self.uniqueY:
                 Ycurrent = np.where(np.array(self.Y)==i)[0]
                 Ystands = np.array(self.stand)[Ycurrent]
                 Ystand = np.unique(Ystands)
                 
                 np.random.seed(self.seed)
                 selectedStand = np.random.permutation(Ystand)[0]
                 

                 if self.SLOO is True:

                     YinSelectedStandt = np.in1d(Ystands,selectedStand)
                     YinSelectedStand = Ycurrent[YinSelectedStandt]
                     validation = np.concatenate((validation,np.asarray(YinSelectedStand)))
                     
                     # improve code...
                     #Ycurrent[sp.where(Ystands!=selectedStand)[0]]

                     YnotInSelectedStandt = np.invert(YinSelectedStandt)
                     YnotInSelectedStand = Ycurrent[YnotInSelectedStandt]
                     train = np.concatenate((train,np.asarray(YnotInSelectedStand)))
                     StandToRemove.append(selectedStand)
                 else:
                     np.random.seed(self.seed)
                     randomYstand = np.random.permutation(Ystand)
                     
                     Ytrain = np.in1d(Ystands,randomYstand[:int(len(Ystand)*self.split)])
                     Ytrain = Ycurrent[Ytrain]
                     Yvalidation = np.in1d(Ystands,randomYstand[int(len(Ystand)*self.split):])
                     Yvalidation = Ycurrent[Yvalidation]
                     
                     train = np.concatenate((train,np.asarray(Ytrain)))
                     validation = np.concatenate((validation,np.asarray(Yvalidation)))
                     
                 
             self.iterPos +=1 
             self.seed += 1
             return train,validation
         else:
             raise StopIteration()
             