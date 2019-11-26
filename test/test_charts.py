# -*- coding: utf-8 -*-
import unittest

import os
import numpy as np
from museotoolbox import charts
confusion_matrix = np.random.randint(5,20,[5,5])

class TestCharts(unittest.TestCase):
    def test_Plot(self):
        pcm = charts.PlotConfusionMatrix(confusion_matrix)
        pcm.add_text()
        pcm.color_diagonal('RdYlBu')
        pcm.add_x_labels([1,2,3,4,5])
        pcm.add_y_labels(['one','two','three','four','five'])
        pcm.add_mean('mean','mean')
        
    def test_Plot2(self):
        pcm = charts.PlotConfusionMatrix(confusion_matrix,left=0.12,right=.9)
        pcm.add_text()
        pcm.add_x_labels([1,2,3,4,5],position='top',rotation=90)
        pcm.add_f1()
        pcm.add_y_labels(['one','two','three','four','five'])
        pcm.save_to('/tmp/test.pdf')
        os.remove('/tmp/test.pdf')

    def test_Plot_accuracy(self):
        pcm = charts.PlotConfusionMatrix(confusion_matrix,left=0.12,right=.9,cmap='PuRd_r')
        pcm.add_text(thresold=35)
        pcm.add_x_labels([1,2,3,4,5],position='top',rotation=90)
        pcm.add_y_labels(['one','two','three','four','five'])
        pcm.add_accuracy()

if __name__ == "__main__":
    unittest.main()
    