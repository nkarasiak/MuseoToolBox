# -*- coding: utf-8 -*-
import unittest

import os
import tempfile
import numpy as np
from museotoolbox import charts
confusion_matrix = np.random.randint(5,20,[5,5])
confusion_matrix[-1,-1] = 0
confusion_matrix[-1,:] = 0

tmp_dir = tempfile.mkdtemp()

class TestCharts(unittest.TestCase):
    def test_Plot(self):
        for hide_ticks in [True,False]:
            pcm = charts.PlotConfusionMatrix(confusion_matrix)
            pcm.color_diagonal('RdYlBu')
            pcm.add_text()
            pcm.add_x_labels([1,2,3,4,5],rotation=59+hide_ticks,position='bottom')
            pcm.add_y_labels(['one','two','three','four','five'])
            pcm.add_mean('mean','mean',hide_ticks=True)
            
    def test_f1(self):
        pcm = charts.PlotConfusionMatrix(confusion_matrix,left=0.12,right=.9)
        pcm.add_text()
        pcm.add_x_labels([1,2,3,4,5],position='top',rotation=90)
        pcm.add_f1()
        
        pcm.add_y_labels(['one','two','three','four','five'])
        pcm.save_to(os.path.join(tmp_dir,'test.pdf'))
        os.remove(os.path.join(tmp_dir,'test.pdf'))
        
    def test_f1_nonsquarematrix(self):
        pcm = charts.PlotConfusionMatrix(confusion_matrix[:,:-2])

        self.assertRaises(Warning,pcm.add_f1)
        self.assertRaises(Warning,pcm.color_diagonal)
        self.assertRaises(Warning,pcm.add_accuracy)
        

    def test_accuracy(self):
        for rotation in [45,90]:
            pcm = charts.PlotConfusionMatrix(confusion_matrix,left=0.12,right=.9,cmap='PuRd_r')
            pcm.add_text(thresold=35)
            pcm.add_x_labels([1,2,3,4,5],position='top',rotation=90)
            pcm.add_y_labels(['one','two','three','four','five'])
            pcm.add_accuracy()
        
if __name__ == "__main__":
    unittest.main()
    