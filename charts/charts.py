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
featImp = np.load('/tmp/feat.npy')

def plotFeatureImportance(feat,sampleTime=False,pdf=False):
    """
    plot feature importance with sampleTime
    
    """
    bands = ('Blue','Green','Red','Infra red')
    if sampleTime:
        dates = np.loadtxt(sampleTime,dtype=int)
        import datetime as dt
        dts = [dt.datetime.strptime(str(k),'%Y%m%d') for k in dates]
    """
    for k in dates:
        l=dt.datetime.strptime(str(k),'%Y%m%d')
        dts.append(dt.datetime.strftime(l,format('%m/%d')))
    """
    best_feat = feat*100
    
    import seaborn as sns
    #import os      
                
    colorList = sns.color_palette("Paired", 16)
    colorList = ['#3c78d8','#93c47d','#dd7e6b','#593535','#ad0000','#8f0000','#000000']
    #colorList = sns.diverging_palette(10, 220, sep=80, n=10)
      
    
    n_dates = len(feat)
    #ind = np.arange(n_dates)    # the x locations for the groups
    #ind = [dt.datetime.strptime(str(k),'%Y%m%d').strftime('%j') for k in dates]

    width = 5      # the width of the bars: can also be len(x) sequence
    #colorList = plt.get_cmap('jet',30)
    
    plt.figure()
    plt.style.use('dark_background')
    plotedBand = []
    """
    for i in range(4):
        B = current_feat[range(i,len(dates)*4,4)]
        if i == 0:
            Bplot = B
            print Bplot
            p1 = plt.bar(ind, B, width, color=colorList[i])#yerr=menStd)    
        elif i>0:        
            bottom=Bplot
            
            p1 = plt.bar(ind, B, width, bottom=bottom,color=colorList[i])#yerr=menStd)        
            Bplot += B

        plotedBand.append(p1)
    
    """   
    #current_feat = current_feat/(sp.sum(current_feat)/100)
    
    sum_by_date = []
    for dateind, i in enumerate(range(0,len(feat))):
        j = [i,i+(len(dates)),i+(len(dates)*2),i+(len(dates)*3)]
        #B = current_feat[j]
        
        for k,l in enumerate(j):
            B = current_feat[l]
            #print Bplot
            if k==0:
                Bplot = B
                p1 = plt.bar(sp.int16(ind[dateind]), B, width, color=colorList[k])#yerr=menStd)    
            else:
                bottom = Bplot
                
                p1 = plt.bar(sp.int16(ind[dateind]), B, width, bottom=bottom,color=colorList[k])#yerr=menStd)    
                
                Bplot +=B   
                
            plotedBand.append(p1)
        sum_by_date.append(sp.sum(current_feat[j]))
        
    
    #BName=('B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12')
    #BName = str(bands)
    BName=bands
    plt.legend((plotedBand[::-1]),BName[::-1],columnspacing=0,labelspacing=0,fontsize=6)
    
    plt.grid(True,linestyle='-',alpha=.5)
    #menStd = (2, 3, 4, 1, 2)
    #womenStd = (3, 5, 2, 3, 3)
    
    
    #p1 = plt.bar(ind, B1, width, color='#d62728')#yerr=menStd)
    #p2 = plt.bar(ind, B2, width,bottom=B1)#yerr=womenStd)
    
    plt.ylabel('\% of importance')
    plt.title('Feature importances for {} in {} with Random Forest'.format(specie,y))
    plt.xticks(sp.int16(ind), dts,rotation=90,horizontalalignment="left",verticalalignment='baseline',size=6)
    import datetime

    maxBar = sp.around(sp.amax(sum_by_date),0)+1
    plt.xlim(0,367)
    plt.ylim(ymin=0,ymax=maxBar+1)
    plt.vlines(int(datetime.datetime.strptime('{}0321'.format(y),'%Y%m%d').strftime('%j')),0,maxBar,colors='white',alpha=.8)
    plt.vlines(int(datetime.datetime.strptime('{}0621'.format(y),'%Y%m%d').strftime('%j')),0,maxBar,colors='white',alpha=.8)
    plt.vlines(int(datetime.datetime.strptime('{}0921'.format(y),'%Y%m%d').strftime('%j')),0,maxBar,colors='white',alpha=.8)
    plt.vlines(int(datetime.datetime.strptime('{}1221'.format(y),'%Y%m%d').strftime('%j')),0,maxBar,colors='white',alpha=.8)
    #plt.vlines(int(datetime.datetime.strptime('{}1221'.format(y),'%Y%m%d').strftime('%j')),0,maxBar)
    #plt.axis('off')
    
    plt.text(int(datetime.datetime.strptime('{}0321'.format(y),'%Y%m%d').strftime('%j'))+1,maxBar,'Spring')
    plt.text(int(datetime.datetime.strptime('{}0621'.format(y),'%Y%m%d').strftime('%j'))+1,maxBar,'Summer')
    plt.text(int(datetime.datetime.strptime('{}0921'.format(y),'%Y%m%d').strftime('%j'))+1,maxBar,'Autumn')
    plt.text(int(datetime.datetime.strptime('{}0120'.format(y),'%Y%m%d').strftime('%j'))+1,maxBar,'Winter')
    
            
    #plt.yticks(np.arange(0, 81, 10))
    #plt.legend((p1[0], p2[0]), ('Men', 'Women'))
    plt.grid(True,alpha=.1)
    pp.savefig(bbox_inches='tight',pad_inches=0.1)
                #pp.sa#fig.savefig('kappaEvo.pdf')
                #pp.close

def plotConfusionMatrix(cm,title='',ylabels=False,xlabels=False,vmin=False,vmax=False,xlabel=False,ylabel=False,\
                 thresold=False,alpha=False,legend=False,percent=False,xhorizontalalignment=False,cmap=plt.cm.Oranges,lastCorner=False,cm2=False,cmap2=plt.cm.Greens,xlabelsPos="bottom",diag=False,\
                 suptitle=False,xrotation=False,thresold2=False,vmin2=False,vmax2=False,dpi=False,toround=0,figsize=(6,6),subplot=False,font=False,fontsize=12,\
                 beautifulBorder=False,pdf=False):
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
        dpi = 150
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
            
        gs = gridspec.GridSpec(2,2, height_ratios=[1,1/cm.shape[1]], width_ratios=[cm.shape[0],1])

        gs.update(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=0.1, hspace=0.1)

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

    fig = plt.figure(1, figsize=figsize,dpi=dpi,tight_layout=True)
    
    if cm2 is not False:
        
        ax = plt.subplot(gs[0,0]) # place it where it should be.

        
        pl1 = ax.imshow(cm,interpolation='nearest',aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha)
        
        if not vmin2:
            vmin2 = np.amin(cm2)
        if not vmax2:
            vmax2 = np.amax(cm2)
        pl2 = ax.imshow(cm2,interpolation='nearest',aspect='auto',cmap=cmap2,vmin=vmin2,vmax=vmax2,alpha=alpha)
        

    else:
        fig = plt.subplot(gs[0,0])
        fig = plt.imshow(cm, interpolation='nearest', aspect='auto',cmap=cmap,vmin=vmin, vmax=vmax,alpha=alpha)
    
    ax.set_title(title,fontsize=fontsize+2)

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

        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
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
                    ax.text(j, i, numberToPlot,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresold else "#888888" if cm[i,j] == 0 else "#444444",va='center')
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
    if xlabel:
        ax.xlabel(xlabel)
    if ylabel :
        ax.ylabel(ylabel)
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
    subplotCmap = plt.cm.Greens
    if subplot:
        ax1v = plt.subplot(gs[0,1])
        # --------------------------------------------------------
        if subplot == 'F1':
            
            # Plot the data
            verticalBarTitle = 'F1'
            verticalPlot = []
        
            for label in range(cm_.shape[1]):
                TP = cm_[label,label]
                #TN = np.sum(sp.diag(currentCsv))-currentCsv[label,label]
                FN = np.sum(cm_[:,label])-TP
                FP = np.sum(cm_[label,:])-TP
            
                verticalPlot.append(2*TP / (2*TP+FP+FN)*100)
            
        #ax1v.hist(y,bins=bins, orientation='horizontal', color='k', edgecolor='w')
        elif subplot == 'Mean':
            verticalBarTitle = 'Mean'
            verticalPlot = [np.mean(cm_,axis=1)]
        else:
            verticalBarTitle = 'User\'s acc.'
            verticalPlot = [np.diag(cm_)/np.sum(cm_,axis=1)*100]
        verticalPlot = np.asarray(verticalPlot).reshape(-1,1)
        ax1v.imshow(verticalPlot,cmap=subplotCmap,interpolation='nearest',aspect='auto',alpha=.8,vmin=0,vmax=100)
        
        ax1v.set_title(verticalBarTitle)    
        # Define the limits, labels, ticks as required
        #ax1v.set_yticks(np.linspace(-4,4,9)) # Ensures we have the same ticks as the scatter plot !
        #ax1v.set_xticklabels([])
        ax1v.set_yticks([],[])
        ax1v.set_xticks([],[])
        
        if beautifulBorder:
            ax1v.tick_params(which="minor", bottom=False, top=False,left=False)

            for edge, spine in ax1v.spines.items():
                spine.set_visible(False)

            ax1v.grid(which="minor", color="w", linestyle='-', linewidth=2)
            ax1v.set_yticks(np.arange(cm.shape[1]+1)-.5, minor=True)
        for i in range(cm_.shape[1]):
            print(i)
            ax1v.text(0,i,int(verticalPlot[i,0]),horizontalalignment="center",color="white" if verticalPlot[i,0] > thresold else "#444444",va='center')

        if subplot is True or subplot is not 'F1':
            ax1h = plt.subplot(gs[1,0])
            print(subplot)
            # --------------------------------------------------------
            if subplot == 'Mean':
                horizontalBar = 'Mean'
                horizontalPlot = np.mean(cm_,axis=0)
            else:
                horizontalBar = 'Prod\'s acc.'
                horizontalPlot = np.diag(cm_)/np.sum(cm_,axis=0)*100
            
            horizontalPlot = horizontalPlot.reshape(1,-1)
            #ax1h.hist(x, bins=bins, orientation='vertical', color='k', edgecolor='w')
            ax1h.imshow(horizontalPlot,cmap=subplotCmap,interpolation='nearest', aspect='auto',alpha=.8,vmin=0,vmax=100)
            # Define the limits, labels, ticks as required
            #ax1h.set_xticks(np.linspace(-4,4,9)) # Ensures we have the same ticks as the scatter plot !
            #ax1h.set_xlim([-4,4])
            #ax1h.set_xlabel(r'My x label')

            
            
            ax1h.set_yticks([0])
            ax1h.set_yticklabels([horizontalBar])
            
            for i in range(cm.shape[0]):
                ax1h.text(i,0,int(horizontalPlot[0,i]),horizontalalignment="center",color="white" if horizontalPlot[0,i] > thresold else "#444444",va='center')
            if xlabelsPos != 'top':
                ax1h.set_xticks(range(cm.shape[0]))
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
    
                ax1h.grid(which="minor", color="w", linestyle='-', linewidth=2)
                ax1h.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
                

    if pdf : 
        plt.savefig(pdf, bbox_inches='tight')
    return fig


if __name__ == '__main__':
    plotFeatureImportance(featImp)