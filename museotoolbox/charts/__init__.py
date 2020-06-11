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

# for numpy version < 1.17


def _nan_to_num(array, nan=0):
    return np.where(np.isnan(array), nan, array)


class PlotConfusionMatrix:
    """
    Plot a confusion matrix with imshow of pyplot.
    Customize color (e.g. diagonal color), add subplots with F1 or Producer/User accuracy.

    Examples
    --------
    >>> plot = mtb.charts.plotConfusionMatrix([[5,6],[1,8]])
    >>> plot.add_text()
    >>> plot.add_f1()
    """

    def __init__(self, cm, cmap=plt.cm.Greens,
                 left=None, right=None, zero_is_min=True, max_is_max=True, **kwargs):
        self.cm = np.array(cm)
        self.cm_ = np.copy(cm)
        self.axes = []

        # init gridspec
        self._left_grisdspec = left
        self._right_grisdspec = right
        self._init_gridspec()

        self.ax = plt.subplot(self.gs[0, 0])  # place it where it should be.
        self.zero_is_min = zero_is_min

        if zero_is_min is True:
            self.vmin = 0
        else:
            self.vmin = np.amin(self.cm)

        if max_is_max is True:
            self.vmax = np.amax(self.cm)
        else:
            self.vmax = max_is_max

        self.xlabelsPos = 'bottom'
        self.xrotation = 0
        self.yrotation = 0
        self.font_size = False

        self.cmap = cmap
        self.diag_color = cmap
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
        self.subplot_ax1v = False
        self.axes.append(self.ax)

    def _init_gridspec(self):
        self.gs = gridspec.GridSpec(
            2, 3, width_ratios=[
                self.cm.shape[1], 1, 1], height_ratios=[
                self.cm.shape[0], 1])

        self.gs.update(
            bottom=0,
            top=1,
            wspace=0,
            hspace=0.7 / self.cm.shape[0],
            right=self._right_grisdspec,
            left=self._left_grisdspec)

    def add_label(self, x_label=False, y_label=False,x_position='top'):
        self.ax.set(xlabel=x_label, ylabel=y_label)
        self.ax.xaxis.set_label_position(x_position) 

    def add_text(self, thresold=False, font_size=12, alpha=1, alpha_zero=1):
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
        >>> plot.add_text(alpha_zero=0.5)
        """
        plt.rcParams.update({'font.size': font_size})

        self.font_size = font_size
        if thresold is False:
            thresold = int(np.amax(self.cm) / 2)
        for i, j in itertools.product(
                range(self.cm.shape[0]), range(self.cm.shape[1])):
            cm_value = self.cm[i, j]
            txt_displayed = str(cm_value) if isinstance(cm_value, (int, np.integer)) \
                else '{:.1f}'.format(cm_value)
            if not np.ma.is_masked(cm_value):
                # print(cm[i,j])
                self.ax.text(j,
                             i,
                             txt_displayed,
                             horizontalalignment="center",
                             color="white" if cm_value > thresold else 'black',
                             fontsize=font_size,
                             va='center',
                             alpha=alpha_zero if cm_value == 0 else alpha)
            else:
                #                print(self.cm2[i, j])
                self.ax.text(j,
                             i,
                             txt_displayed,
                             horizontalalignment="center",
                             color="white" if cm_value > thresold else "black",
                             va='center',
                             fontsize=font_size,
                             )

    def add_x_labels(self, labels=None, rotation=90, position='top'):
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
        >>> plot.add_text(labels=['Tofu','Houmous'],alpha_zero=0.5,rotation=45)
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

    def add_mean(self, xLabel='', yLabel='', hide_ticks=False,
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
        >>> plot.add_mean(xLabel='all species',yLabel='all years')
        """

        if self.subplot_ax1v is not False:
            self._init_gridspec()
            if self.subplot_ax1v == 'F1':
                self.subplot_ax1v = 'Mean'
                self.add_f1()
        self.subplot_ax1v = 'Mean'

        self.ax1v = plt.subplot(self.gs[0, 1])
        self.ax1h = plt.subplot(self.gs[1, 0])

        valV = np.mean(self.cm_, axis=1).reshape(-1, 1).astype(int)

        valH = np.mean(self.cm_, axis=0).reshape(1, -1).astype(int)

        self.ax1v.imshow(
            valV,
            cmap=self.diag_color,
            interpolation='nearest',
            aspect='equal',
            vmin=vmin,
            vmax=vmax)
        self.ax1h.imshow(
            valH,
            cmap=self.diag_color,
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

            iVal = np.int(np.mean(self.cm_, axis=1)[i])

            self.ax1v.text(
                0,
                i,
                iVal,
                color="white" if iVal > thresold else 'black',
                ha='center',
                va='center',
                fontsize=self.font_size)

        self.ax1v.set_yticklabels([])
        for j in range(self.cm.shape[1]):
            jVal = np.int(np.mean(self.cm_, axis=0)[j])

            self.ax1h.text(
                j,
                0,
                jVal,
                color="white" if jVal > thresold else 'black',
                ha='center',
                va='center',
                fontsize=self.font_size)

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

    def add_y_labels(self, labels=None, rotation=0):
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
        >>> plot.add_y_labels(labels=['Fried','Raw'])
        """
        self.yrotation = rotation
        self.ylabels = labels
        self.ax.set_yticklabels(
            self.ylabels,
            rotation=rotation,
            horizontalalignment='right',
            fontsize=self.font_size)

    def add_f1(self):
        """
        Add F1 subplot.

        Examples
        --------
        >>> plot.add_f1()
        """
        if self.cm.shape[0] != self.cm.shape[1]:
            raise Warning('Number of lines and columns must be equal')

        if self.subplot_ax1v is False or self.subplot_ax1v == 'F1':
            self.ax1v = plt.subplot(self.gs[0, 1])
            current_ax = self.ax1v
            self.subplot_ax1v = 'F1'
        else:
            self.ax2v = plt.subplot(self.gs[0, 2])
            current_ax = self.ax2v

        verticalPlot = []

        for label in range(self.cm.shape[0]):
            TP = self.cm_[label, label]
            #TN = np.sum(sp.diag(currentCsv))-currentCsv[label,label]
            FN = np.nansum(self.cm_[:, label]) - TP
            FP = np.nansum(self.cm_[label, :]) - TP

            verticalPlot.append(2 * TP / (2 * TP + FP + FN) * 100)

        if self.font_size is not False:
            font_size = self.font_size
        else:
            font_size = 12

        verticalPlot = np.asarray(verticalPlot).reshape(-1, 1)
        current_ax.imshow(
            verticalPlot,
            cmap=self.diag_color,
            interpolation='nearest',
            aspect='equal',
            vmin=0,
            vmax=100)

        if self.xlabelsPos == 'top':
            current_ax.xaxis.tick_top()
            current_ax.xaxis.set_ticks_position(
                'top')  # THIS IS THE ONLY CHANGE
            current_ax.set_xticks([0])
            current_ax.set_xticklabels(
                ['F1'],
                horizontalalignment='center',
                rotation=self.xrotation,
                size=font_size)
        else:
            current_ax.set_xticks([0])
            current_ax.set_xticklabels(
                ['F1'],
                horizontalalignment='left',
                rotation=self.xrotation,
                size=font_size)
        current_ax.set_yticks([])

        for i in range(self.cm.shape[0]):
            txt = str(int(_nan_to_num(verticalPlot[i])))

            current_ax.text(
                0,
                i,
                txt,
                size=font_size,
                horizontalalignment="center",
                color="white" if verticalPlot[i] > 50 else "black",
                va='center')
        self.axes.append(current_ax)

    def add_accuracy(self, thresold=50, invert_PA_UA=False,
                     user_acc_label='User\'s acc.',
                     prod_acc_label='Prod\'s acc.'):
        """
        Add user and producer accuracy.

        Parameters
        ----------
        thresold : int, default 50
            The thresold value where text will be in white instead of black.
        invert_PA_UA : bool, default False
            If True, user and producer accuracy labels are switched (note that
            it does not reverse the confusion matrix though).
        user_acc_label: str
            The user accuracy label to display. Defautl to 'User's acc.'
        prod_acc_label: str
            The user accuracy label to display. Defautl to 'Prod's acc.'

        Examples
        --------
        >>> plot.add_accuracy()
        """

        if self.font_size is not False:
            font_size = self.font_size
        else:
            font_size = 12

        if self.subplot_ax1v is not False:
            self._init_gridspec()
            
            if self.subplot_ax1v == 'F1':
                self.subplot_ax1v = 'accuracy'
                self.add_f1()
        self.subplot_ax1v = 'accuracy'

        if self.cm_.shape[0] != self.cm_.shape[1]:
            raise Warning('Number of lines and columns must be equal')

        self.ax1v = plt.subplot(self.gs[0, 1])
        self.ax1h = plt.subplot(self.gs[1, 0])

        self.ax1v.imshow(np.array(np.diag(self.cm_) / np.nansum(self.cm_,
                                                                axis=1) * 100).reshape(-1,
                                                                                       1),
                         cmap=self.diag_color,
                         interpolation='nearest',
                         aspect='equal',
                         vmin=0,
                         vmax=100)

        self.ax1h.imshow(np.array(np.diag(self.cm_) / np.nansum(self.cm_, axis=0) * 100).reshape(
            1, -1), cmap=self.diag_color, interpolation='nearest', aspect='equal', vmin=0, vmax=100)

        self.ax1v.set_yticks(np.arange(self.cm_.shape[0]))
        self.ax1v.set_xticks([])

        self.ax1h.set_yticks([0])
        self.ax1h.set_xticks([])

        for i in range(self.cm.shape[0]):
            iVal = np.int(_nan_to_num(np.array(np.diag(
                self.cm_) / np.nansum(self.cm_, axis=1) * 100).reshape(-1, 1)[i][0], nan=0))

            self.ax1v.text(
                0,
                i,
                iVal,
                color="white" if iVal > thresold else 'black',
                size=font_size,
                ha='center',
                va='center')

        self.ax1v.set_yticklabels([])
        for j in range(self.cm.shape[1]):
            jVal = np.int(_nan_to_num(np.array(np.diag(
                self.cm_) / np.nansum(self.cm_, axis=0) * 100).reshape(-1, 1)[j][0], nan=0))

            self.ax1h.text(
                j,
                0,
                jVal,
                color="white" if jVal > thresold else 'black',
                size=font_size,
                ha='center',
                va='center')

        y_label, x_label = [prod_acc_label], [user_acc_label]
        if invert_PA_UA:
            x_label, y_label = y_label, x_label

        self.ax1h.set_yticklabels(
            y_label,
            rotation=self.yrotation,
            ha='right',
            va='center',
            size=font_size)

        self.ax1v.xaxis.set_ticks_position('top')  # THIS IS THE ONLY CHANGE
        self.ax1v.set_xticks([0])
        if self.xrotation < 60:
            ha = 'left'
        else:
            ha = 'center'
        self.ax1v.set_xticklabels(
            x_label,
            horizontalalignment='left',
            rotation=self.xrotation,
            ha=ha,
            size=font_size)
        # TOFIX not extend ?
        self.axes.append([self.ax1v, self.ax1h])

    def color_diagonal(self, diag_color=plt.cm.Greens,
                       matrix_color=plt.cm.Reds):
        """
        Add user and producer accuracy.

        Parameters
        ----------
        diag_color : pyplot colormap, default plt.cm.Greens.
        matrix_color : pyplot colormap, default plt.cm.Reds

        Examples
        --------
        >>> plot.colorDiag()
        """
        self.diag_color = diag_color
        if self.cm.shape[0] != self.cm.shape[1]:
            raise Warning(
                'Array must have the same number of lines and columns')

        mask = np.zeros(self.cm.shape)
        np.fill_diagonal(mask, 1)

        self.cm2 = np.ma.masked_array(self.cm, mask=np.logical_not(mask))
        self.cm = np.ma.masked_array(self.cm, mask=mask)
        if self.zero_is_min is True:
            vmin = 0
        else:
            vmin = np.amin(
                self.cm_),

        self.ax.imshow(
            self.cm2,
            interpolation='nearest',
            aspect='equal',
            cmap=diag_color,
            vmin=vmin,
            vmax=np.amax(
                self.cm_),
            alpha=1)
        self.ax.imshow(
            self.cm,
            interpolation='nearest',
            aspect='equal',
            cmap=matrix_color,
            vmin=vmin,
            vmax=np.amax(
                self.cm_),
            alpha=1)

    def save_to(self, path, dpi=150):
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
