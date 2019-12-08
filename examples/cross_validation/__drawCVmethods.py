#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 10:41:59 2019

@author: nicolas
"""
import numpy as np
from matplotlib import pyplot as plt

def plotMethod(cvType='SKF-pixel'):
    nSamples = 30
    alpha_unused = 0.1
    trainColor='C0'
    validColor='C1'
    points = np.array([]).reshape(-1,3)
    distanceBuffer=200
    bufferSize=48000
    #random
    np.random.seed(10)
    def multiplyBy100AndToInt(x):
        x*=100
        x=x.astype(int)
        return x
    def initFrame(lim=200,size=4):
        f=plt.figure(figsize=(size,size))
        plt.xlim(-lim,lim)
        plt.ylim(-lim,lim)
        plt.xticks([])
        plt.yticks([])
        return f
        
    def drawFrame(title,train,valid,unused=False,buffer=False,show=True):
        f=initFrame()
        ax = f.add_subplot(111)
    
        """
        if title:
            plt.title(title)
        """
        ax.scatter(train[0],train[1],c=trainColor,marker='.',s=100,label='Train')
        ax.scatter(valid[0],valid[1],c=validColor,marker='x',label='Valid')
        ax.legend()
        if unused is not False:
            ax.scatter(unused[0],unused[1],c='grey',marker='.',s=100,alpha=alpha_unused,label='Unused references')
        if buffer is not False:
            ax.scatter(valid[0],valid[1],facecolor='none',edgecolor='red',marker='o',s=bufferSize)
        
        if show:
            plt.show()
        return f
    
    #### Generate 4 stands
            
    X = np.random.vonmises(5,50,nSamples)
    Y = np.random.vonmises(5,10,nSamples)
    X = multiplyBy100AndToInt(X)
    Y = multiplyBy100AndToInt(Y)
    
    label = np.ones(Y.shape)
    points = np.concatenate((points,np.vstack((X,Y,label)).T))
    # plt.scatter(X,Y)
    
    
    X = np.random.vonmises(0.3,7,nSamples)-1
    Y = np.random.vonmises(0,10,nSamples)+1
    X = multiplyBy100AndToInt(X)
    Y = multiplyBy100AndToInt(Y)
    
    label += 1
    points = np.concatenate((points,np.vstack((X,Y,label)).T))
    
    # plt.scatter(X,Y)
    X = np.random.vonmises(1,8,nSamples)
    Y = np.random.vonmises(1,20,nSamples)
    X = multiplyBy100AndToInt(X)
    Y = multiplyBy100AndToInt(Y)
    label += 1
    points = np.concatenate((points,np.vstack((X,Y,label)).T))
    
    X = np.random.vonmises(100,15,nSamples)+1
    Y = np.random.vonmises(100,10,nSamples)-0.3
    X = multiplyBy100AndToInt(X)
    Y = multiplyBy100AndToInt(Y)
    label += 1
    points = np.concatenate((points,np.vstack((X,Y,label)).T))
      
    paths = []
    codes = []
    
    from scipy.spatial import ConvexHull
    import matplotlib.path as mpath
    
    codes += [mpath.Path.MOVETO] + \
                     (len(X)-1)*[mpath.Path.LINETO]
    
    for group in range(1,5):
        coords = points[np.where(points[:,2]==group)][:,:2]
        h=ConvexHull(coords).vertices
    
        path = mpath.Path(coords[h])
        paths.append(path)
            
    randomPoints = np.random.permutation(points)
    
    if cvType == 'SKF-pixel':
        f=drawFrame('Stratified-K-Fold pixel-based',[randomPoints[:,0][:75],randomPoints[:,1][:75]],[randomPoints[:,0][75:],randomPoints[:,1][75:]])
        plt.show()
    else:
        train = np.where(np.in1d(points[:,2],np.array([2,4])))
        valid = np.where(np.in1d(points[:,2],np.array([2,4]),invert=True))
    
        if cvType == 'SKF-group':
            f=drawFrame('Stratified-K-Fold group-based',[points[valid][:,0],points[valid][:,1]],[points[train][:,0],points[train][:,1]])
            plt.show()

        else:
            valid = randomPoints[0]
            train = randomPoints[1:]
            if cvType == 'LOO-pixel':
                f=drawFrame('Leave-One-Out pixel-based',[train[:,0],train[:,1]],valid)
                
            if cvType == 'LOO-group':
                valid = np.where(np.in1d(points[:,2],np.array([4])))
                train = np.where(np.in1d(points[:,2],np.array([4]),invert=True))
                f=drawFrame('Leave-One-Out group-based',[points[train][:,0],points[train][:,1]],[points[valid][:,0],points[valid][:,1]])
#                pp.savefig(f,bbox_inches='tight')
                # SLOO-pixel
            if cvType == 'SLOO-pixel':
                from scipy.spatial import distance
                distance=distance.cdist(randomPoints[:,:2],randomPoints[:,:2])
                
                valid = randomPoints[0]
                train_nospatial = randomPoints[1:]
                train = randomPoints[np.where(distance[0,:]>distanceBuffer)[0]]
                f=drawFrame('Spatial Leave-One-Out pixel-based',[train[:,0],train[:,1]],valid,[train_nospatial[:,0],train_nospatial[:,1]],buffer=True,show=False)
                plt.text(-70,0,'Distance buffer\nfrom validation pixel',fontsize=12)
                #lt.scatter(train_nospatial[:,0],train_nospatial[:,1],c='grey',marker='.',s=100,alpha=alpha_unused)
                plt.show()
            
            # 
            if cvType == 'SLOO-group':
                valid = np.where(np.in1d(points[:,2],np.array([4])))
                train_nospatial = np.where(np.in1d(points[:,2],np.array([4]),invert=True))
                
                train = np.where(np.in1d(points[:,2],np.array([2])))
                
                train=[points[train][:,0],points[train][:,1]]
                valid=[points[valid][:,0],points[valid][:,1]]
                unused=points[train_nospatial][:,0],points[train_nospatial][:,1]
                f=drawFrame('Spatial Leave-One-Out group-based',train,valid,unused,show=False)
                
                centroid = np.asarray([[np.mean(points[:,0][np.where(points[:,2]==stand)]),np.mean(points[:,1][np.where(points[:,2]==stand)])] for stand in range(1,5)])
                plt.scatter(centroid[:,0],centroid[:,1],color='black',s=60,alpha=0.8,label='Centroid')
                plt.scatter(centroid[:,0][3],centroid[:,1][3],facecolor='none',edgecolor='red',marker='o',s=bufferSize)
                plt.text(-90,-10,'Distance buffer\nfrom centroid',fontsize=12)
                plt.show()
            