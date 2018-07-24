# coding: utf-8
from scipy import stats
import numpy as np
import sys
sys.path.append("..")
from tools import rasterTools

class computeClassificationStability():
        def __init__(self,inRaster,outRaster,inMaskRaster,outGdalDT,outNoData):
            #process = rasterTools.readAndWriteRaster(inRaster,outRaster=outRaster,inMaskRaster=inMaskRaster,outNBand=2,outGdalGDT=outGdalDT,outNoData=outNoData)
            process = rasterTools.readAndWriteRaster(inRaster,inMaskRaster)
            process.addFunction(self.stabCalc,outRaster,2,3,outNoData)
            process.run()
            
        def stabCalc(self,arr): 
            tmp = stats.mode(arr,axis=-1)
            tmpStack = np.column_stack((tmp[0],tmp[1]))
            return tmpStack
        
#process.iterProcessAndWrite(returnArr)

if __name__ == '__main__':
    inRaster = "/mnt/DATA/Formosat_2006-2014/v2/classification/SVM/SLOO_meanAllYearFromMedian/level3/10.vrt"
    #inRaster = "/mnt/DATA/Sentinel-2/2017/5days/SITS_forestMask.tif"
    outRaster = "/mnt/DATA/Formosat_2006-2014/v2/classification/SVM/SLOO_meanAllYearFromMedian/modal.tif"
    inMaskRaster = False# "/mnt/DATA/Formosat_2006-2014/v2/data/forestMask.tif"
    
    computeClassificationStability(inRaster,outRaster,inMaskRaster,outGdalDT=3,outNoData =0)
    