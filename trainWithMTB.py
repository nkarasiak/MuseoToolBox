#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 10:12:14 2018

@author: nicolas
"""
import sys
import tempfile
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import MuseoToolBox as mtb

if os.uname()[1] == 'karasiak-HPEliteBook':
    inVector = '/mnt/DATA/Projets/Autocorrelation/Data/vector/dep34.gpkg'
    inRaster = '/mnt/DATA/Projets/Autocorrelation/Data/raster/SITS.tif'
    resultDir = '/mnt/DATA/Projets/Autocorrelation/results/'
elif os.uname()[1] == 'TOSCA':
    inVector = '/media/10T/autocorrelation/Data/vector/dep34.gpkg'
    inRaster = None
    resultDir = '/media/10T/autocorrelation/results/'


# Settings Number of stands AND CV methods
try:
    nStandsToTest = list(map(int,sys.argv[2].split(',')))
except IndexError:
    nStandsToTest = [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        1972]

try:
    slooTypesArg = list(sys.argv[1].split(','))
    if slooTypesArg == ['all']:
        slooTypes = ['RS50', 'LOO', 'RS100', 'SLOO']
    else:
        slooTypes = slooTypesArg
except IndexError :
    slooTypes = ['RS50', 'LOO', 'RS100', 'SLOO']

ROIprefix = 'band_'

coordsFile = os.path.join(resultDir + 'coords.npy')
if not os.path.exists(coordsFile):
    ROItemp = tempfile.mktemp('roi.tif')
    ROItemp = mtb.rasterTools.rasterize(inRaster, inVector, 'level', ROItemp)
    coords = mtb.rasterTools.getSamplesFromROI(
        inRaster, ROItemp, getCoords=True, onlyCoords=True)
    np.save(coordsFile, coords)
    os.remove(ROItemp)
    #distanceArray = vectorTools.distMatrix(coords)
else:
    coords = np.load(coordsFile)
    #distanceArray = vectorTools.distMatrix(coords)

noExt = os.path.abspath(inVector).split('.')[0]
if not os.path.exists(noExt + '_x.npy'):
    Y, S, D, X = mtb.vectorTools.readValuesFromVector(
        inVector, 'level', 'stand', 'dep', bandPrefix='band_')
    removeNDVIabove04 = np.where(
        (X[:, 3] - X[:, 2]) / (X[:, 3] + X[:, 2]) > 0.4)[0]
    X = X[removeNDVIabove04, :]
    Y = Y[removeNDVIabove04]
    S = S[removeNDVIabove04]
    D = D[removeNDVIabove04]
    np.save(noExt + '_x.npy', X)
    np.save(noExt + '_ysd.npy', [Y, S, D])

    if coords.shape[0] != Y.shape[0]:
        coords = coords[removeNDVIabove04, :]
        np.save(coordsFile, coords)

else:
    X = np.load(noExt + '_x.npy')
    Y, S, D = np.load(noExt + '_ysd.npy')

#from sklearn.model_selection import train_test_split
extFile = noExt[:-2] + '81_xTest.npy'


distStand = os.path.join(os.path.dirname(inVector),'distPerStand.npz')
if 'SLOO' in slooTypes:    
    if not os.path.exists(distStand):
        if os.uname()[1] == 'karasiak-HPEliteBook':
            YY = mtb.vectorTools.readValuesFromVector('/mnt/DATA/Projets/Autocorrelation/Data/vector/centroid_34.gpkg','stand')
            # SY if Group label in the distanceMatrix
            #YS = YY[np.in1d(YY,np.unique(S))]
            Sdistance,Ys = mtb.vectorTools.getDistanceMatrix(inRaster,'/mnt/DATA/Projets/Autocorrelation/Data/vector/centroid_34.gpkg','stand')
            SY = Ys[np.in1d(Ys,np.unique(S))]
            Sdistance = Sdistance[np.where(np.in1d(Ys,np.unique(S))==True)[0],:][:,np.where(np.in1d(Ys,np.unique(S))==True)[0]]
            np.savez(distStand,Sdistance=Sdistance.astype(np.int16),SY=SY.astype(np.int32))
        else:
            raise Exception('Please, generate the distance matrix per stand using the raster')



def predictExternal(outModel, Xexternal, Yexternal):
    LAP = mtb.learnAndPredict(-1)
    LAP.loadModel(outModel)
    X_pred = LAP.predictFromArray(Xexternal)
    confusionMatrix = confusion_matrix(Yexternal, X_pred)
    np.savetxt(outModel + '.csv', confusionMatrix, delimiter=',', fmt='%.d')


def parseModel(name,cv,group=None):
    # init
    classifier = RandomForestClassifier(n_jobs=-1,class_weight=['balanced'])
    param_grid = dict(n_estimators=[200], class_weight=['balanced'])
    saveDir = os.path.join(resultDir, name, str(nStandPerClass), str(idx))
    outModel = saveDir + '_rf.npy'
    if name=='RS100': outStatsFromCV=False
    else: outStatsFromCV=True
    
    print('X shape is '+str(curX.shape))
    print('y shape is '+str(curY.shape))
    print('group shape is '+str(group.shape))
    print('unique group is  '+str(np.unique(group,return_counts=True)))
    
    if not os.path.exists(outModel):
        print(outModel)
        # init LearnAndPredict
        LAP = mtb.learnAndPredict(n_jobs=-1)

        LAP.learnFromVector(
            curX,
            curY,
            group,
            classifier=classifier,
            param_grid=param_grid,
            outStatsFromCV=outStatsFromCV,
            scale=True,
            cv=cv)

        # save results
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        # save each CM
        if name!='RS100':
            for statsidx,cmAndNtrain in enumerate(LAP.getStatsFromCV(nTrain=True)):
                np.savetxt(
                    saveDir +
                    '/' +
                    str(statsidx) +
                    '.csv',
                    cmAndNtrain[0],
                    header="Training samples : "+','.join(str(tr) for tr in cmAndNtrain[1]),
                    fmt='%0.d')
                
        LAP.saveModel(outModel)

for nStandPerClass in nStandsToTest:
    for idx in range(10):
        print(str(nStandPerClass) + '/' + str(idx))
        np.random.seed(idx)

        # generate random stands, random seed is set to repeat later the
        # operation
        
        curX, curY, curGroup, curCoords = np.array(
            [[], [], [], []], dtype=np.int64)
        curX = curX.reshape(-1, X.shape[1]).astype(np.float)
        curCoords = curCoords.reshape(-1, 2).astype(np.int64)
        for i in np.unique(Y):
            level = np.where(Y == i)[0]
            levelstand = np.random.permutation(np.unique(S[np.where(Y == i)[0]]))[
                :nStandPerClass]
            levelstandTF = np.in1d(S, levelstand)

            curX = np.vstack((curX, X[levelstandTF, :]))
            curY = np.concatenate(
                (curY, np.full(X[levelstandTF, :].shape[0], i)))
            curGroup = np.concatenate((curGroup, S[levelstandTF]))
            curCoords = np.concatenate((curCoords, coords[levelstandTF]))

        if 'RS100' in slooTypes:            
            cv = None
            parseModel('RS100',cv=cv)

        if 'RS50' in slooTypes:
            RS50 = mtb.crossValidationTools.RandomCV  # RS50
            cv = RS50(n_splits=10, random_state=idx)
            parseModel('RS50pixel',cv=cv,group=None)

            LPSGO = mtb.crossValidationTools.LeavePSubGroupOut  # RS50 group
            cv = LPSGO(n_splits=10, random_state=idx)
            parseModel('RS50group',cv=cv,group=curGroup)

        if 'LOO' in slooTypes:
            if nStandPerClass < 10:
                # LOO is too much resource consumption to iter above 10 stands
                from sklearn.model_selection import LeaveOneOut
                cv = LeaveOneOut()
                parseModel('LOOpixel',cv=cv,group=None)

            LOSGO = mtb.crossValidationTools.LeaveOneSubGroupOut  # LOO group
            cv = LOSGO()
            parseModel('LOOgroup',cv=cv,group=curGroup)

        if 'SLOO' in slooTypes:
            if nStandPerClass < 100:
                from scipy.spatial import distance
                distanceMatrix = distance.cdist(
                    curCoords, curCoords).astype(
                            np.int64)
                # SLOPO
                """
		SLOPO = mtb.crossValidationTools.SpatialLeaveOnePixelOut  # SLOPO
                cv = SLOPO(
                    distanceThresold=1964,
                    distanceMatrix=distanceMatrix,
                    random_state=idx)
                parseModel('SLOOpixel',cv=cv)
                """
                try:
                    SLOSGO = mtb.crossValidationTools.SpatialLeaveOneSubGroupOut # SLOSGO
                    tmp = np.load(distStand)
                    Sdistance = tmp['Sdistance']
                    Slabel = tmp['SY']
                    groupIdxToKeep = np.where(np.in1d(Slabel,np.unique(curGroup))==1)[0]
                    distanceMatrixGroup = Sdistance[groupIdxToKeep,:][:,groupIdxToKeep]
                    distanceLabelGroup = Slabel[groupIdxToKeep]
                    cv = SLOSGO(
                            distanceThresold=1964,
                            distanceMatrix=distanceMatrixGroup,
                            distanceLabel=distanceLabelGroup,
                            random_state=idx)
                    parseModel('SLOOgroup',cv=cv,group=curGroup)
                    
                    del distanceLabelGroup,distanceMatrixGroup,curGroup,Slabel
                except ValueError:
                    print('Not enought distance between stands')
