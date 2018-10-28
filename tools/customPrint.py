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
from __future__ import absolute_import, print_function
import sys

def pushFeedback(message,feedback=None):
    """
    pushFeedback, manage Qgis feedback in QgsProcessing
    
    Parameters
    -------
    message : str, int, float.
        If str, will print the message everywhere.
        If int/float, only show in the feedback progressbar.
    """
    isNum = isinstance(message,(float,int))
    
    if feedback and feedback is not True:
        if feedback=='gui':
            if not isNum:
                QgsMessageLog.logMessage(str(message))
        else:
            if isNum:
                feedback.setProgress(message)
            else:
                feedback.setProgressText(message)
    else:
        if not isNum:
            print(str(message))            

class progressBar:
    def __init__(self,total,message='',length=50):
        """
        total : int
            Total number of samples.
        length : int.
            Length of the bar.
        """
        self.start = 0
        self.total = total
        self.length = length
        self.message = message
        self.lastPosition = None
        
    def addPosition(self,value):
        inPercent = int(value/self.total*100)
        if inPercent != self.lastPosition :
            self.lastPosition = inPercent
            self.nHash = int(self.length*(value/self.total))
            self.nPoints = int(self.length-int(self.nHash))
            self.printBar(inPercent)
        
    def printBar(self,value):
        if value == 100:
            end = "\n"
        else:
            end = "\r"
        sys.stdout.flush()
        #print(self.nHash)
        #print(self.nPoints)
        print(self.message+' [{}{}]{}%'.format(self.nHash*"#",self.nPoints*".",self.lastPosition),end=end)
    
if __name__ == '__main__':
    pb = progressBar(800,length=50)
    import time
    for i in [100,300,500,700,800]:
        pb.addPosition(i)
        time.sleep(1)

        
            
