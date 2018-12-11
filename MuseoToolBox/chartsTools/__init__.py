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

from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import itertools


class plotConfusionMatrix:
    def __init__(self,cm,cmap=plt.cm.Greens,**kwargs):      
        self.cm = np.array(cm)
        self.cm_ = np.copy(cm)
        self.axes = []
        self.gs = gridspec.GridSpec(2,2,width_ratios=[self.cm.shape[0],1],height_ratios=[self.cm.shape[1], 1])
        self.gs.update(bottom=0,top=1, wspace=0.05/self.cm.shape[0], hspace=0.7/self.cm.shape[1])
        
        self.ax = plt.subplot(self.gs[0,0]) # place it where it should be.
        self.vmin = np.amin(self.cm)
        self.vmax = np.amax(self.cm)
         
        self.xlabelsPos = 'bottom'
        self.xrotation = 0
        self.yrotation = 0
        
        self.ax.set_yticks(range(self.cm.shape[0]))
        
        self.fig = plt.figure(1,tight_layout=True)
        self.ax.imshow(cm,interpolation='nearest', aspect='equal',cmap=cmap,vmin=self.vmin,vmax=self.vmax)
        
        self.kwargs = kwargs
        self.subplot = False
        self.axes.append(self.ax)
    def addText(self,thresold=False,alpha=1,alpha_zero=1):
        if thresold is False:
            thresold = int(np.amax(self.cm)/2)
        for i, j in itertools.product(range(self.cm.shape[0]), range(self.cm.shape[1])):   
            if not np.ma.is_masked(self.cm[i,j]):
                #print(cm[i,j])
                self.ax.text(j, i, str(self.cm[i,j]),
                 horizontalalignment="center",
                 color="white" if self.cm[i, j] > thresold else 'black',va='center',alpha=alpha_zero if self.cm[i,j] == 0 else alpha)
            else:
                print(self.cm2[i,j])
                self.ax.text(j, i, str(self.cm2[i,j]),
                 horizontalalignment="center",
                 color="white" if self.cm2[i, j] > thresold else "black",va='center')
                
    def addXlabels(self,labels=None,rotation=0,position='top')  :
        self.xrotation=rotation
        self.xlabels = labels
        self.xlabelsPos = position
        if self.xlabelsPos == 'top':
            self.ax.xaxis.tick_top()
            self.ax.xaxis.set_ticks_position('top') # THIS IS THE ONLY CHANGE
            
        self.ax.set_xticklabels(['F1'],horizontalalignment='left',rotation=rotation)

        self.ax.set_xticks(np.arange(self.cm.shape[1]))
        self.ax.set_xticklabels(self.xlabels,rotation=rotation,horizontalalignment='center')

    def addYlabels(self,labels=None,rotation=0):
        self.yrotation=rotation
        self.ylabels=labels
        self.ax.set_yticklabels(self.ylabels,rotation=rotation,horizontalalignment='right')
        
    def addF1(self):
        if self.subplot is not False:
            raise Warning('You can\'t add two subplots. You already had '+str(self.subplot))
        elif self.cm.shape[0] != self.cm.shape[1]:
            raise Warning('Number of lines and columns must be equal')
        else: self.subplot='F1'
        self.ax1v = plt.subplot(self.gs[0,1])
    
        verticalPlot = []
    
        for label in range(self.cm.shape[0]):
            TP = self.cm_[label,label]
            #TN = np.sum(sp.diag(currentCsv))-currentCsv[label,label]
            FN = np.sum(self.cm_[:,label])-TP
            FP = np.sum(self.cm_[label,:])-TP
        
            verticalPlot.append(2*TP / (2*TP+FP+FN)*100)
        
        verticalPlot = np.asarray(verticalPlot).reshape(-1,1)
        self.ax1v.imshow(verticalPlot,cmap=plt.cm.Greens,interpolation='nearest',aspect='equal',vmin=0,vmax=100)
        
        if self.xlabelsPos == 'top':
            self.ax1v.xaxis.tick_top()
            self.ax1v.xaxis.set_ticks_position('top') # THIS IS THE ONLY CHANGE
            self.ax1v.set_xticks([0])
            self.ax1v.set_xticklabels(['F1'],horizontalalignment='center',rotation=self.xrotation)
        else:
            self.ax1v.set_xticks([0])
            self.ax1v.set_xticklabels(['F1'],horizontalalignment='left',rotation=self.xrotation)
        self.ax1v.set_yticks([])
        
        for i in range(self.cm.shape[0]):
            try:
                txt = str(int(verticalPlot[i]))
            except:
                txt= '0'
            
            self.ax1v.text(0,i,txt,horizontalalignment="center",color="white" if verticalPlot[i] > 50 else "black",va='center')
        self.axes.append(self.ax1v)
    def colorDiag(self,matrixCmap=plt.cm.Greens,diagCmap=plt.cm.Reds):
        
        if self.cm.shape[0] != self.cm.shape[1]:
            raise Exception('Array must have the same number of lines and columns')
            
        mask = np.zeros(self.cm.shape)
        np.fill_diagonal(mask,1)

        self.cm2 = np.ma.masked_array(self.cm,mask=np.logical_not(mask))
        self.cm = np.ma.masked_array(self.cm,mask=mask)
        
        self.ax.imshow(self.cm2,interpolation='nearest',aspect='equal',cmap=matrixCmap,vmin=np.amin(self.cm_),vmax=np.amax(self.cm_),alpha=1)
        self.ax.imshow(self.cm,interpolation='nearest',aspect='equal',cmap=diagCmap,vmin=np.amin(self.cm_),vmax=np.amax(self.cm_),alpha=1)
    
    def addAccuracy(self,thresold=50):
        
        if self.subplot is not False:
            raise Warning('You can\'t add two subplots. You already had '+str(self.subplot))
        elif self.cm_.shape[0] != self.cm_.shape[1]:
            raise Warning('Number of lines and columns must be equal')
        else: self.subplot='F1'
        
        self.ax1v = plt.subplot(self.gs[0,1])
        self.ax1h = plt.subplot(self.gs[1,0])
        
        self.ax1v.imshow(np.array(np.diag(self.cm_)/np.sum(self.cm_,axis=1)*100).reshape(-1,1),cmap=plt.cm.Greens,interpolation='nearest',aspect='equal',vmin=0,vmax=100)
        self.ax1h.imshow(np.array(np.diag(self.cm_)/np.sum(self.cm_,axis=0)*100).reshape(1,-1),cmap=plt.cm.Greens,interpolation='nearest',aspect='equal',vmin=0,vmax=100)
        
        
        self.ax1v.set_yticks(np.arange(self.cm_.shape[0]))
        self.ax1v.set_xticks([])
        
        
        self.ax1h.set_yticks([0])
        self.ax1h.set_xticks([])
        
        for i in range(self.cm.shape[0]):
            try:
                iVal = np.int(np.array(np.diag(self.cm_)/np.sum(self.cm_,axis=1)*100).reshape(-1,1)[i][0])
            except:
                iVal = 0
            self.ax1v.text(0,i,iVal,color="white" if iVal>thresold else 'black',ha='center')
                
        self.ax1v.set_yticklabels([])
        for j in range(self.cm.shape[1]):
            try:
                jVal = np.int(np.array(np.diag(self.cm_)/np.sum(self.cm_,axis=0)*100).reshape(-1,1)[j][0])
            except:
                jVal = 0 
            self.ax1h.text(j,0,jVal,color="white" if jVal>thresold else 'black',ha='center')
        
        self.ax1h.set_yticklabels(['Prod\'s acc.'],rotation=self.yrotation,ha='right',va='center')
        
        
        self.ax1v.xaxis.set_ticks_position('top') # THIS IS THE ONLY CHANGE
        self.ax1v.set_xticks([0])
        self.ax1v.set_xticklabels(['User\'s acc.'],horizontalalignment='left',rotation=self.xrotation,ha='center')
        self.axes.append([self.ax1v,self.ax1h])
        
    def show(self):
        plt.show(self.fig)
        
    def saveTo(self,path,dpi=150):
        self.fig.savefig(path,dpi=dpi,bbox_inches='tight')
        
    def setWhiteBorders(self):
        def __removeLines(ax):
            ax.tick_params(which="minor", bottom=False, top=False,left=False)

            for edge, spine in ax.spines.items():
                spine.set_visible(False)
    
            ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
            
            ax.set_xticks(np.arange(len(ax.get_xticks())+1)-.49, minor=True)
            ax.set_yticks(np.arange(len(ax.get_yticks())+1)-.5, minor=True)
            
        for ax in self.axes:
            __removeLines(ax)

