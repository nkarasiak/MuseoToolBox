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
try:
    from matplotlib import pyplot as plt
    import itertools
except Exception as error:
    raise ImportError(error)
   
def plotConfusionMatrix(cm,title='',ylabels=False,xlabels=False,vmin=False,vmax=False,xlabel=False,ylabel=False,\
                 thresold=False,alpha=False,legend=False,percent=False,xhorizontalalignment=False,cmap=plt.cm.Oranges,lastCorner=False,cm2=False,cmap2=plt.cm.Greens,xlabelsPos="bottom",diag=False,\
                 suptitle=False,xrotation=False,thresold2=False,vmin2=False,vmax2=False,dpi=False,toround=0,figsize=(6,6),subplot=False,font=False,fontsize=12,\
                 beautifulBorder=False,aspect='auto',fmt=False,fmtSub=False,subPlotVerticalTitle=False,subplotCmap=False,subPlotHorizontalTitle=False,gsright=.95,pdf=False):
    """
    plot Confusion matrix, with lot (too many ?) of custom available.
    
    Parameters
    ----------
    cm : array
        confusion matrix
    title : str0000
    subplot : str in list
        ['F1','Mean',True]
    """
    cm_ = np.copy(cm)
    if dpi is False:
        dpi = 80

    if font or fontsize:
        try:
            from matplotlib import rcParams
        except Exception as error:
            raise ImportError(error)
        rcParams['font.family'] = 'sans-serif'
        if font :
            rcParams['font.sans-serif'] = [font]
        if fontsize:
            rcParams['font.size'] = fontsize
        
    if thresold is False:
        thresold = np.mean(cm[cm>0])
        thresold2 = np.mean(np.diag(cm))
    if subplot:
        try:
            from matplotlib import gridspec
        except Exception as error:
            raise ImportError(error)
            
        gs = gridspec.GridSpec(2,2, height_ratios=[cm.shape[0],1], width_ratios=[cm.shape[1],1])

        gs.update(left=0.05, right=gsright, bottom=0.08, top=0.92, wspace=0.1, hspace=0.1)

    if not alpha:
        alpha = 1.0
    if diag is not False:
        mask = np.zeros(cm.shape)
        np.fill_diagonal(mask,1)

        cm2 = np.ma.masked_array(cm,mask=np.logical_not(mask))
        cm = np.ma.masked_array(cm,mask=mask)

    if vmin or vmax:
        if vmin:
            vmin=vmin
        if vmax:
            vmax=vmax
    else:
        if percent:
            vmin=0
            vmax=100
        else:
            vmin=np.amin(cm)
            vmax=np.amax(cm)

    fig = plt.figure(1, figsize=figsize,dpi=dpi)#,tight_layout=True)

    if cm2 is not False or subplot:
        if subplot is not False:
            ax = plt.subplot(gs[0,0]) # place it where it should be.

        
        pl1 = ax.imshow(cm,interpolation='nearest', aspect=aspect,cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha)
        
        if not vmin2:
            vmin2 = np.amin(cm2)
        if not vmax2:
            vmax2 = np.amax(cm2)
        if cm2 is not False:
            pl2 = ax.imshow(cm2,interpolation='nearest',aspect=aspect,cmap=cmap2,vmin=vmin2,vmax=vmax2,alpha=alpha)
        

    else:
        

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow(cm, interpolation='nearest', aspect=aspect,cmap=cmap,vmin=vmin, vmax=vmax,alpha=alpha)

    ax.set_title(title,fontsize=fontsize+2)
    #ax.set_adjustable('box-forced')


    ax = plt.gca()
    
    ax.set_yticks(range(cm.shape[0]))
    
    if ylabels:
        ax.set_yticklabels(ylabels,rotation=0,fontsize=fontsize,horizontalalignment='right')

    if not xhorizontalalignment:
        xhorizontalalignment = "center"
    if xlabels and subplot is False or subplot is 'F1' or xlabelsPos is 'top':
        #plt.xticks(range(len(xlabels)))
        
        if xlabelsPos=='top':        
            ax.xaxis.tick_top()
            ax.xaxis.set_ticks_position('top') # THIS IS THE ONLY CHANGE
            
            rotation=270
        else:
            rotation=4
        if xrotation:
            rotation=xrotation
            ax.set_xticklabels(xlabels,rotation=rotation,horizontalalignment=xhorizontalalignment)
        else:
            ax.set_xticklabels(xlabels,horizontalalignment=xhorizontalalignment)
        ax.set_xticks(range(len(xlabels)))
    else:
        ax.set_xticks([],[])
        #ax.set_xticklabels((ax.get_xticks() +1).astype(str))            
    if beautifulBorder:
        """
        ax.tick_params(top=True, bottom=False,
               labeltop=True, labelbottom=False)
        """
        ax.tick_params(which="minor", bottom=False, top=False,left=False)

        for edge, spine in ax.spines.items():
            spine.set_visible(False)

        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
            
    
        
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):   
        if not np.isnan(cm[i,j]):
            #print(cm[i,j])
            if not np.ma.is_masked(cm[i,j]):
                
                if percent:
                    if toround:
                        numberToPlot = str(round(cm[i, j],toround))+'%'
                    else:
                        numberToPlot = str(int(cm[i, j]))+'%'
                else:
                    if toround:
                        numberToPlot = str(round(cm[i, j],toround))
                    else:
                        numberToPlot = str(int(cm[i, j]))
                    if numberToPlot == '0.0':
                        numberToPlot = '0'
                if lastCorner and i==cm.shape[0]-1 and j==cm.shape[1]-1:
                    ax.text(j,i,'')
                else:
                    if fmt:numberToPlot=str(numberToPlot+fmt)
                    
                    ax.text(j, i, str(numberToPlot),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresold else "#888888" if cm[i,j] < 0 else "#444444",va='center')
            if cm2 is not False:
                if not np.ma.is_masked(cm2[i,j]):
                    if percent:
                        numberToPlot = str(round(cm2[i, j],toround))+'%'
                    else:
                        if toround:
                            numberToPlot = str(round(cm2[i, j],toround))
                        else:
                            numberToPlot = str(int(cm2[i, j]))
                        if numberToPlot == '0.0':
                            numberToPlot = '0'
                    
                    if lastCorner and i==cm.shape[0]-1 and j==cm.shape[1]-1:
                        ax.text(j,i,'')
                    else:
                        ax.text(j, i, numberToPlot,
                         horizontalalignment="center",
                         color="white" if cm2[i,j] > thresold2 else "#444444",va='center')
                
            
    #plt.ylabel('True label')

    if legend is True:
        if cm2 is not False:
            fig.colorbar(pl1,ax=ax)
            fig.colorbar(pl2,ax=ax)
        else:
            plt.colorbar()
    if suptitle:
        plt.suptitle(suptitle)
    """
    tick_marks = np.arange(cm.shape[1])
    
    if not xlabels:
        plt.xticks(tick_marks, [],rotation=45)
    """
    #plt.tight_layout()
    if xlabel:
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(xlabel)  
        #plt.xlabel(xlabel)
    if ylabel :
        plt.ylabel(ylabel)
    if subplotCmap is False:
        subplotCmap = plt.cm.Greens
    if subplot:
        ax1v = plt.subplot(gs[0,1])
        # --------------------------------------------------------
        if subplot == 'F1':
            
            # Plot the data
            subPlotVerticalTitle = 'F1'
            verticalPlot = []
        
            for label in range(cm_.shape[1]):
                TP = cm_[label,label]
                #TN = np.sum(sp.diag(currentCsv))-currentCsv[label,label]
                FN = np.sum(cm_[:,label])-TP
                FP = np.sum(cm_[label,:])-TP
            
                verticalPlot.append(2*TP / (2*TP+FP+FN)*100)
            
        #ax1v.hist(y,bins=bins, orientation='horizontal', color='k', edgecolor='w')
        elif subplot == 'Mean':
            if subPlotVerticalTitle is False:
                subPlotVerticalTitle = 'Mean'
            verticalPlot = [np.mean(cm_,axis=1)]
        else:
            subPlotVerticalTitle = 'User\'s acc.'
            verticalPlot = [np.diag(cm_)/np.sum(cm_,axis=1)*100]
        verticalPlot = np.asarray(verticalPlot).reshape(-1,1)
        ax1v.imshow(verticalPlot,cmap=subplotCmap,interpolation='nearest', aspect=aspect,alpha=.8,vmin=0,vmax=100)
        
        #ax1v.set_title(verticalBarTitle)    
        # Define the limits, labels, ticks as required
        #ax1v.set_yticks(np.linspace(-4,4,9)) # Ensures we have the same ticks as the scatter plot !
        #ax1v.set_xticklabels([])
        if xlabelsPos == 'top':
            ax1v.xaxis.tick_top()
            ax1v.xaxis.set_ticks_position('top') # THIS IS THE ONLY CHANGE
            
            rotation=270
            
            ax1v.set_xticks([0])
            ax1v.set_xticklabels([subPlotVerticalTitle],horizontalalignment=xhorizontalalignment,rotation=xrotation)
        else:
            ax1v.set_xticklabels([subPlotVerticalTitle],horizontalalignment=xhorizontalalignment,rotation=xrotation)
            ax1v.set_xticks([])
        ax1v.set_yticks([])
        #ax1v.set_xticks([0],['co'])
        #ax1v.set_xticks([],[])
        


        for i in range(cm_.shape[0]):
            txt = str(int(verticalPlot[i,0]))
            if fmtSub: txt += fmtSub
            ax1v.text(0,i,txt,horizontalalignment="center",color="white" if verticalPlot[i,0] > thresold else "#444444",va='center')
        if beautifulBorder:
            ax1v.tick_params(which="minor", bottom=False, top=False,left=False)

            for edge, spine in ax1v.spines.items():
                spine.set_visible(False)

            ax1v.grid(which="minor", color="w", linestyle='-', linewidth=1)
            ax1v.set_yticks(np.arange(cm.shape[1]+1)-.5, minor=True)
            
        if subplot is True or subplot is not 'F1':
            ax1h = plt.subplot(gs[1,0])
            print(subplot)
            # --------------------------------------------------------
            if subplot == 'Mean':
                if not subPlotHorizontalTitle:
                    subPlotHorizontalTitle = 'Mean'
                horizontalPlot = np.mean(cm_,axis=0)
            else:
                subPlotHorizontalTitle = 'Prod\'s acc.'
                horizontalPlot = np.diag(cm_)/np.sum(cm_,axis=0)*100
            
            horizontalPlot = horizontalPlot.reshape(1,-1)
            #ax1h.hist(x, bins=bins, orientation='vertical', color='k', edgecolor='w')
            ax1h.imshow(horizontalPlot,cmap=subplotCmap,interpolation='nearest', aspect=aspect,alpha=.8,vmin=0,vmax=100)
            # Define the limits, labels, ticks as required
            #ax1h.set_xticks(np.linspace(-4,4,9)) # Ensures we have the same ticks as the scatter plot !
            #ax1h.set_xlim([-4,4])
            #ax1h.set_xlabel(r'My x label')

            #ax1h.set_title(horizontalBar)
            
            ax1h.set_yticks([0])
            ax1h.set_yticklabels([subPlotHorizontalTitle])
            
            for i in range(cm.shape[1]):
                txt = str(int(horizontalPlot[0,i]))
                if fmtSub: txt += fmtSub
                ax1h.text(i,0,txt,horizontalalignment="center",color="white" if horizontalPlot[0,i] > thresold else "#444444",va='center')
            if xlabelsPos != 'top':
                ax1h.set_xticks(range(cm.shape[1]))
                if xrotation:
                    ax1h.set_xticklabels(xlabels,ha=xhorizontalalignment,rotation=xrotation)
                else:
                    ax1h.set_xticklabels(xlabels,ha=xhorizontalalignment)
            else:
                ax1h.set_xticks([],[])
            
            
            if beautifulBorder:
                ax1h.tick_params(which="minor", bottom=False, top=False,left=False)
    
                for edge, spine in ax1h.spines.items():
                    spine.set_visible(False)
    
                ax1h.grid(which="minor", color="w", linestyle='-', linewidth=1)
                ax1h.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    if pdf:
        fig.savefig(pdf,bbox_inches='tight')

    return fig