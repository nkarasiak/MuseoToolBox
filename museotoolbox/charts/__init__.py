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
# @git:     www.github.com/nkarasiak/MuseoToolBox
# =============================================================================
"""
The :mod:`museotoolbox.charts` module gathers plotting functions.
"""

from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import itertools

np.seterr(divide='ignore', invalid='ignore')


class plotConfusionMatrix:
    """
    Plot a confusion matrix with imshow of pyplot.
    Customize color (e.g. diagonal color), add subplots with F1 or Producer/User accuracy.

    Examples
    --------
    >>> plot = mtb.charts.plotConfusionMatrix([[5,6],[1,8]])
    >>> plot.addText()
    >>> plot.addF1()
    """

    def __init__(self, cm, cmap=plt.cm.Greens,
                 left=None, right=None, **kwargs):
        self.cm = np.array(cm)
        self.cm_ = np.copy(cm)
        self.axes = []
        self.gs = gridspec.GridSpec(
            2, 2, width_ratios=[
                self.cm.shape[1], 1], height_ratios=[
                self.cm.shape[0], 1])

        self.gs.update(
            bottom=0,
            top=1,
            wspace=0,
            hspace=0.7 / self.cm.shape[0], right=right, left=left)

        self.ax = plt.subplot(self.gs[0, 0])  # place it where it should be.
        self.vmin = np.amin(self.cm)
        self.vmax = np.amax(self.cm)

        self.xlabelsPos = 'bottom'
        self.xrotation = 0
        self.yrotation = 0
        self.font_size = False

        self.cmap = cmap
        self.diagColor = cmap
        self.ax.set_yticks(range(self.cm.shape[0]))

        self.fig = plt.figure(1)

        self.ax.imshow(
            cm,
            interpolation='nearest',
            aspect='equal',
            cmap=self.cmap,
            vmin=self.vmin,
            vmax=self.vmax,
            **kwargs)

        self.kwargs = kwargs
        self.subplot = False
        self.axes.append(self.ax)

    def addText(self, thresold=False, font_size=12, alpha=1, alpha_zero=1):
        """
        Add value of each case on the matrix image.

        Parameters
        ----------
        thresold : False or integer.
        alpha : float, default 1.
            Value from 0 to 1.
        alpha_zero : float, default 1.
            Value alpha for 0 values, from 0 to 1.

        Examples
        --------
        >>> plot.addText(alpha_zero=0.5)
        """
        self.font_size = font_size
        if thresold is False:
            thresold = int(np.amax(self.cm) / 2)
        for i, j in itertools.product(
                range(self.cm.shape[0]), range(self.cm.shape[1])):
            if not np.ma.is_masked(self.cm[i, j]):
                # print(cm[i,j])
                self.ax.text(j, i, str(self.cm[i, j]),
                             horizontalalignment="center",
                             color="white" if self.cm[i, j] > thresold else 'black', fontsize=font_size, va='center', alpha=alpha_zero if self.cm[i, j] == 0 else alpha)
            else:
                #                print(self.cm2[i, j])
                self.ax.text(j, i, str(self.cm2[i, j]),
                             horizontalalignment="center",
                             color="white" if self.cm2[i, j] > thresold else "black", va='center', fontsize=font_size, )

    def addXlabels(self, labels=None, rotation=90, position='top'):
        """
        Add labels for X.

        Parameters
        ----------
        labels : None
            If labels, best with same len as the X shape.
        rotation : int, default 90.
            Int, 45 or 90 is best.
        position : str, default 'top'.
            'top' or 'bottom'.

        Examples
        --------
        >>> plot.addText(labels=['Tofu','Houmous'],alpha_zero=0.5,rotation=45)
        """
        self.xrotation = rotation
        self.xlabels = labels
        self.xlabelsPos = position
        if self.xlabelsPos == 'top':
            self.ax.xaxis.tick_top()
            self.ax.xaxis.set_ticks_position('top')  # THIS IS THE ONLY CHANGE

        self.ax.set_xticklabels(
            ['F1'],
            horizontalalignment='left',
            rotation=rotation, fontsize=self.font_size)

        self.ax.set_xticks(np.arange(self.cm.shape[1]))
        if rotation != 90:
            ha = 'left'
        else:
            ha = 'center'
        self.ax.set_xticklabels(
            self.xlabels,
            rotation=rotation,
            ha=ha,
            fontsize=self.font_size)

    def addMean(self, xLabel='', yLabel='', hide_ticks=False,
                thresold=50, vmin=0, vmax=100):
        """
        Add Mean for both axis.

        Parameters
        ----------
        xLabel : str
            The label for X (i.e. 'All species')
        yLabel : str
            The label for Y (i.e. 'All years')
        thresold : int, default 50.
        vmin : int.
            Minimum value for colormap.
        vmax :
            Maximum value for colormap.

        Examples
        --------
        >>> plot.addMean(xLabel='all species',yLabel='all years')
        """
        if self.subplot is not False:
            raise Warning(
                'You can\'t add two subplots. You already had ' + str(self.subplot))
        else:
            self.subplot = 'Mean'

        self.ax1v = plt.subplot(self.gs[0, 1])
        self.ax1h = plt.subplot(self.gs[1, 0])

        try:
            valV = np.mean(self.cm_, axis=1).reshape(-1, 1).astype(int)
        except BaseException:
            valV = 0

        try:
            valH = np.mean(self.cm_, axis=0).reshape(1, -1).astype(int)
        except BaseException:
            valH = 0

        self.ax1v.imshow(
            valV,
            cmap=self.diagColor,
            interpolation='nearest',
            aspect='equal',
            vmin=vmin,
            vmax=vmax)
        self.ax1h.imshow(
            valH,
            cmap=self.diagColor,
            interpolation='nearest',
            aspect='equal',
            vmin=vmin,
            vmax=vmax)

        if hide_ticks:
            self.ax1v.set_yticks([])
        else:
            self.ax1v.set_yticks(np.arange(self.cm_.shape[0]))
        self.ax1v.set_xticks([])

        self.ax1h.set_yticks([0])
        self.ax1h.set_xticks([])

        for i in range(self.cm.shape[0]):
            try:
                iVal = np.int(np.mean(self.cm_, axis=1)[i])
            except BaseException:
                iVal = 0
            self.ax1v.text(0, i, iVal, color="white" if iVal >
                           thresold else 'black', ha='center', va='center', fontsize=self.font_size)

        self.ax1v.set_yticklabels([])
        for j in range(self.cm.shape[1]):
            try:
                jVal = np.int(np.mean(self.cm_, axis=0)[j])
            except BaseException:
                jVal = 0
            self.ax1h.text(j, 0, jVal, color="white" if jVal >
                           thresold else 'black', ha='center', va='center', fontsize=self.font_size)

        self.ax1h.set_yticklabels(
            [yLabel],
            rotation=self.yrotation,
            ha='right',
            va='center', fontsize=self.font_size)

        self.ax1v.xaxis.set_ticks_position('top')  # THIS IS THE ONLY CHANGE
        self.ax1v.set_xticks([0])
        if self.xrotation < 60:
            ha = 'left'
        else:
            ha = 'center'
        self.ax1v.set_xticklabels(
            [xLabel],
            horizontalalignment='left',
            rotation=self.xrotation,
            ha=ha,
            fontsize=self.font_size)
        self.axes.append([self.ax1v, self.ax1h])

    def addYlabels(self, labels=None, rotation=0):
        """
        Add labels for Y.

        Parameters
        ----------
        labels : None
            If labels, best with same len as the X shape.
        rotation : int, default 90.
            Int, 45 or 90 is best.

        Examples
        --------
        >>> plot.addYlabels(labels=['Fried','Raw'])
        """
        self.yrotation = rotation
        self.ylabels = labels
        self.ax.set_yticklabels(
            self.ylabels,
            rotation=rotation,
            horizontalalignment='right',
            fontsize=self.font_size)

    def addF1(self):
        """
        Add F1 subplot.

        Examples
        --------
        >>> plot.addF1()
        """
        if self.subplot is not False:
            raise Warning(
                'You can\'t add two subplots. You already had ' + str(self.subplot))
        elif self.cm.shape[0] != self.cm.shape[1]:
            raise Warning('Number of lines and columns must be equal')
        else:
            self.subplot = 'F1'
        self.ax1v = plt.subplot(self.gs[0, 1])

        verticalPlot = []

        for label in range(self.cm.shape[0]):
            TP = self.cm_[label, label]
            #TN = np.sum(sp.diag(currentCsv))-currentCsv[label,label]
            FN = np.sum(self.cm_[:, label]) - TP
            FP = np.sum(self.cm_[label, :]) - TP

            verticalPlot.append(2 * TP / (2 * TP + FP + FN) * 100)

        verticalPlot = np.asarray(verticalPlot).reshape(-1, 1)
        self.ax1v.imshow(
            verticalPlot,
            cmap=self.diagColor,
            interpolation='nearest',
            aspect='equal',
            vmin=0,
            vmax=100)

        if self.xlabelsPos == 'top':
            self.ax1v.xaxis.tick_top()
            self.ax1v.xaxis.set_ticks_position(
                'top')  # THIS IS THE ONLY CHANGE
            self.ax1v.set_xticks([0])
            self.ax1v.set_xticklabels(
                ['F1'],
                horizontalalignment='center',
                rotation=self.xrotation)
        else:
            self.ax1v.set_xticks([0])
            self.ax1v.set_xticklabels(
                ['F1'],
                horizontalalignment='left',
                rotation=self.xrotation)
        self.ax1v.set_yticks([])

        for i in range(self.cm.shape[0]):
            try:
                txt = str(int(verticalPlot[i]))
            except BaseException:
                txt = '0'

            self.ax1v.text(
                0,
                i,
                txt,
                horizontalalignment="center",
                color="white" if verticalPlot[i] > 50 else "black",
                va='center')
        self.axes.append(self.ax1v)

    def addAccuracy(self, thresold=50):
        """
        Add user and producer accuracy.

        Parameters
        ----------
        thresold : int, default 50
            The thresold value where text will be in white instead of black.

        Examples
        --------
        >>> plot.addAccuracy()
        """
        if self.subplot is not False:
            raise Warning(
                'You can\'t add two subplots. You already had ' + str(self.subplot))
        elif self.cm_.shape[0] != self.cm_.shape[1]:
            raise Warning('Number of lines and columns must be equal')
        else:
            self.subplot = 'F1'

        self.ax1v = plt.subplot(self.gs[0, 1])
        self.ax1h = plt.subplot(self.gs[1, 0])

        self.ax1v.imshow(np.array(np.diag(self.cm_) / np.sum(self.cm_,
                                                             axis=1) * 100).reshape(-1,
                                                                                    1),
                         cmap=self.diagColor,
                         interpolation='nearest',
                         aspect='equal',
                         vmin=0,
                         vmax=100)
        self.ax1h.imshow(np.array(np.diag(self.cm_) / np.sum(self.cm_, axis=0) * 100).reshape(
            1, -1), cmap=self.diagColor, interpolation='nearest', aspect='equal', vmin=0, vmax=100)

        self.ax1v.set_yticks(np.arange(self.cm_.shape[0]))
        self.ax1v.set_xticks([])

        self.ax1h.set_yticks([0])
        self.ax1h.set_xticks([])

        for i in range(self.cm.shape[0]):
            try:
                iVal = np.int(np.array(
                    np.diag(self.cm_) / np.sum(self.cm_, axis=1) * 100).reshape(-1, 1)[i][0])
            except BaseException:
                iVal = 0
            self.ax1v.text(0, i, iVal, color="white" if iVal >
                           thresold else 'black', ha='center', va='center')

        self.ax1v.set_yticklabels([])
        for j in range(self.cm.shape[1]):
            try:
                jVal = np.int(np.array(
                    np.diag(self.cm_) / np.sum(self.cm_, axis=0) * 100).reshape(-1, 1)[j][0])
            except BaseException:
                jVal = 0
            self.ax1h.text(j, 0, jVal, color="white" if jVal >
                           thresold else 'black', ha='center', va='center')

        self.ax1h.set_yticklabels(
            ['Prod\'s acc.'],
            rotation=self.yrotation,
            ha='right',
            va='center')

        self.ax1v.xaxis.set_ticks_position('top')  # THIS IS THE ONLY CHANGE
        self.ax1v.set_xticks([0])
        if self.xrotation < 60:
            ha = 'left'
        else:
            ha = 'center'
        self.ax1v.set_xticklabels(
            ['User\'s acc.'],
            horizontalalignment='left',
            rotation=self.xrotation,
            ha=ha)
        self.axes.append([self.ax1v, self.ax1h])

    def colorDiag(self, diagColor=plt.cm.Greens, matrixColor=plt.cm.Reds):
        """
        Add user and producer accuracy.

        Parameters
        ----------
        diagcolor : pyplot colormap, default plt.cm.Greens.
        matrixColor : pyplot colormap, default plt.cm.Reds

        Examples
        --------
        >>> plot.colorDiag()
        """

        if self.cm.shape[0] != self.cm.shape[1]:
            raise Exception(
                'Array must have the same number of lines and columns')

        mask = np.zeros(self.cm.shape)
        np.fill_diagonal(mask, 1)

        self.cm2 = np.ma.masked_array(self.cm, mask=np.logical_not(mask))
        self.cm = np.ma.masked_array(self.cm, mask=mask)

        self.diagColor = diagColor

        self.ax.imshow(
            self.cm2,
            interpolation='nearest',
            aspect='equal',
            cmap=diagColor,
            vmin=np.amin(
                self.cm_),
            vmax=np.amax(
                self.cm_),
            alpha=1)
        self.ax.imshow(
            self.cm,
            interpolation='nearest',
            aspect='equal',
            cmap=matrixColor,
            vmin=np.amin(
                self.cm_),
            vmax=np.amax(
                self.cm_),
            alpha=1)

    def show(self):
        """
        To force plotting the graph
        """

        plt.show(self.fig)

    def saveTo(self, path, dpi=150):
        """
        Save the plot

        Parameters
        ----------
        path : str
            The path of the file to save.
        dpi : int, default 150.

        Examples
        --------
        >>> plot.saveTo('/tmp/contofu.pdf',dpi=300)
        """
        self.fig.savefig(path, dpi=dpi, bbox_inches='tight')

    def setWhiteBorders(self):
        def __removeLines(ax):
            ax.tick_params(which="minor", bottom=False, top=False, left=False)

            for edge, spine in ax.spines.items():
                spine.set_visible(False)

            ax.grid(which="minor", color="w", linestyle='-', linewidth=2)

            ax.set_xticks(
                np.arange(len(ax.get_xticks()) + 1) - .49, minor=True)
            ax.set_yticks(np.arange(len(ax.get_yticks()) + 1) - .5, minor=True)

        for ax in self.axes:
            __removeLines(ax)
