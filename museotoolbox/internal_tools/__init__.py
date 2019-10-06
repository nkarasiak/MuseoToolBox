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
import sys


def pushFeedback(msg, feedback=None):

    if feedback and feedback is not True:
        if feedback == 'gui':
            QgsMessageLog.logMessage(str(msg))
        else:
            feedback.setProgressText(msg)
    else:
        print(msg)


class progressBar:
    def __init__(self, total, message='', length=40, feedback=None):
        """
        total : int
            Total number of samples.
        message : str
            Custom message to show before the progress bar.
        length : int.
            Length of the bar.
        feedback : str, feedback class fro Qgis, or None.
            if str, only 'gui' to log a message in QgsMessageLog.
            if feedback, class feedback from Qgis in order to set the progress bar direclty in the processing toolbox.
        """
        self.start = 0
        self.total = total
        self.length = length
        self.message = message
        self.lastPosition = None
        self.feedback = feedback

    def addPosition(self, value=False):
        """
        Add progress to the bar.

        Parameters
        ----------

        value : int or False.
            If false, will add one.
        """

        if value is False:
            self.start += 1
            value = self.start
            inPercent = int(self.start / self.total * 100)

        else:
            inPercent = int(value / self.total * 100)
        if inPercent != self.lastPosition:
            self.lastPosition = inPercent
            self.nHash = int(self.length * (value / self.total))
            self.nPoints = int(self.length - int(self.nHash))

            if self.feedback:
                self.feedback.setProgress(self.lastPosition)
            else:
                self.printBar(inPercent)

    def printBar(self, value):
        if value == 100:
            end = "\n"
        else:
            end = "\r"

        # print(self.nHash)
        # print(self.nPoints)
        print('\r' + self.message + ' [{}{}]{}%'.format(self.nHash * "#", self.nPoints * ".", self.lastPosition), end=end, flush=True)


if __name__ == '__main__':
    pb = progressBar(200, length=50)
    import time
    for i in range(100):
        pb.addPosition()
        time.sleep(0.01)
    pb.addPosition(200)
