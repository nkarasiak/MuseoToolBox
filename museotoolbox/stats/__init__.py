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
The :mod:`museotoolbox.stats` module gathers stats functions/classes.
"""
import numpy as np


def commissionOmission(table):
    """
    Compute commission and omission from a confusion matrix

    Parameters
    ----------
    table : array.
        The confusion matrix (same number of lines and columns)

    Returns
    -------
    com : commissions (list)
    om : omission (list)
    """
    com, om = [[], []]
    for i in range(table.shape[0]):
        com.append((np.sum(table[i, :]) - table[i, i]
                    ) / np.sum(table[i, :]) * 100)
        om.append((np.sum(table[:, i]) - table[i, i]) /
                  np.sum(table[:, i]) * 100)
    return com, om


class computeConfusionMatrix:
    def __init__(self, yp, yr, kappa=False, OA=False, F1=False):
        """
        Compute confusion matrix given label predicted and label reality.

        Parameters
        ----------
        yp : array.
            The label predicted.
        yr : array.
            The label truth.
        kappa : bool, default False.
            If true, computes kappa.
        OA : bool, default False.
            If True, computes Overall Accuracy.
        F1 : bool, default False.
            If True, computes F1-Score per class.
        """
        # Initialization
        n = yp.size
        C = np.amax((int(yr.max()), int(yp.max())))
        self.confusion_matrix = np.zeros((C, C), dtype=np.int64)

        # Compute confusion matrix
        for i in range(n):
            self.confusion_matrix[yp[i].astype(
                np.int64) - 1, yr[i].astype(np.int64) - 1] += 1

        # Compute overall accuracy
        if OA:
            self.OA = np.sum(np.diag(self.confusion_matrix)) / n

        # Compute Kappa
        if kappa:
            nl = np.sum(self.confusion_matrix, axis=1)
            nc = np.sum(self.confusion_matrix, axis=0)
            self.Kappa = ((n**2) * np.sum(np.diag(self.confusion_matrix)) / n - np.sum(nc * nl)) / \
                (n**2 - np.sum(nc * nl))

        #
        if F1:
            F1 = []
            for label in range(self.confusion_matrix.shape[0]):
                TP = self.confusion_matrix[label, label]
                #TN = np.sum(sp.diag(currentCsv))-currentCsv[label,label]
                FN = np.sum(
                    self.confusion_matrix[:, label]) - self.confusion_matrix[label, label]
                FP = np.sum(
                    self.confusion_matrix[label, :]) - self.confusion_matrix[label, label]
                denum = (2 * TP + FP + FN)
                if denum != 0:
                    F1.append(2 * TP / (2 * TP + FP + FN))
                self.F1 = F1


class statsFromConfusionMatrix:
    def __init__(self, confusionMatrix):
        """
        Get stats (OA, kappa, F1 and F1 man) from confusion matrix

        """
        self.confusionMatrix = confusionMatrix
        self.n = np.sum(self.confusionMatrix)
        self.OA = self.__get_OA()
        self.kappa = self.__get_kappa()
        self.F1mean = self.__get_F1Mean()
        self.F1 = self.__get_F1()

    def __get_OA(self):
        """
        Compute overall accuracy
        """
        return np.sum(np.diag(self.confusionMatrix)) / float(self.n)

    def __get_kappa(self):
        """
        Compute Kappa
        """
        nl = np.sum(self.confusionMatrix, axis=1)
        nc = np.sum(self.confusionMatrix, axis=0)
        OA = np.sum(np.diag(self.confusionMatrix)) / float(self.n)
        return ((self.n**2) * OA - np.sum(nc * nl)) / \
            (self.n**2 - np.sum(nc * nl))

    def __get_F1Mean(self):
        """
        Compute F1 Mean
        """
        nl = np.sum(self.confusionMatrix, axis=1, dtype=float)
        nc = np.sum(self.confusionMatrix, axis=0, dtype=float)
        return 2 * np.mean(np.divide(np.diag(self.confusionMatrix), (nl + nc)))

    def __get_F1(self):
        """
        Compute F1 per class
        """
        f1 = []
        for label in range(self.confusionMatrix.shape[0]):
            TP = self.confusionMatrix[label, label]
            #TN = np.sum(sp.diag(currentCsv))-currentCsv[label,label]
            FN = np.sum(self.confusionMatrix[:, label]) - \
                self.confusionMatrix[label, label]
            FP = np.sum(self.confusionMatrix[label, :]) - \
                self.confusionMatrix[label, label]

            f1.append(2 * TP / (2 * TP + FP + FN))
        return f1
