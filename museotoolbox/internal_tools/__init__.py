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


def push_feedback(msg, feedback=None):
    # in order to convert in Qgis Processing
    # =============================================================================
    #     if feedback and feedback is not True:
    #         if feedback == 'gui':
    #             QgsMessageLog.logMessage(str(msg))
    #         else:
    #             feedback.setProgressText(msg)
    #     else:
    # =============================================================================
    print(msg)


class ProgressBar:

    def __init__(self, total, message='', length=40):
        """
        total : int
            Total number of samples.
        message : str
            Custom message to show before the progress bar.
        length : int.
            Length of the bar.
        """
        self.start = 0
        self.total = total
        self.length = length
        self.message = message
        self.lastPosition = None

    def add_position(self, value=False):
        """
        Add progress to the bar.

        Parameters
        ----------

        value : int or False.
            If false, will add one.
        """

        if value:
            inPercent = int(value / self.total * 100)
        else:

            self.start += 1
            value = self.start
            inPercent = int(self.start / self.total * 100)

        if inPercent != self.lastPosition:
            self.lastPosition = inPercent
            self.nHash = int(self.length * (value / self.total))
            self.nPoints = int(self.length - int(self.nHash))

            self.printBar(inPercent)

    def printBar(self, value):
        if value == 100:
            end = "\n"
        else:
            end = "\r"

        # print(self.nHash)
        # print(self.nPoints)
        print(
            '\r' +
            self.message +
            ' [{}{}]{}%'.format(
                self.nHash *
                "#",
                self.nPoints *
                ".",
                self.lastPosition),
            end=end,
            flush=True)
