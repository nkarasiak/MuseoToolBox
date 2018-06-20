#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
#   _____   _____ _              _ _    _ _   
#  |  __ \ / ____| |            | | |  (_) |  
#  | |__) | (___ | |_ ___   ___ | | | ___| |_ 
#  |  _  / \___ \| __/ _ \ / _ \| | |/ / | __|
#  | | \ \ ____) | || (_) | (_) | |   <| | |_ 
#  |_|  \_\_____/ \__\___/ \___/|_|_|\_\_|\__|
#                                             
#                                             
# 
# @author:  Nicolas Knkarasiak
# @site:    www.karasiak.net
# @git:     www.github.com/lennepkade/rstoolkit 
# =============================================================================

import os
import numpy as np
import glob
import sys,argparse

def computeSITS(S2Dir,outSITS,resample20mBands=False,resampleCSV=False,unzip=False,OTBPythonBinding=True,\
                nbcore=1,ram=256):
    """
    Compute Satellite Image Time Series from Sentinel-2 A/B.
    
    Parameters
    ----------
    S2Dir : Directory
        Directory where zip files from THEIA L2A are unzipped, or zipped.
    outSITS : Output raster
        Output name of your raster (tif format, int16)
    resample20mBands : Bool, default False.
        If True, resample 20m bands at 10m and compute SITS with 10 bands
    resampleCSV : str, default False.
        If str, must be csv file with one line per date (YYYYMMDD format, i.e 20180223).
    unzip : Bool, default False.
        If True, unzip only mandatory images, plus xml and jpg thumbnails.
    OTBPythonBinding : Bool, default True.
        If True, use OTB python binding to avoid creating VRT.
    nbcore : int, default 1.
        Number of cores used.
    ram : int, default 256
        Available ram in mb.
    """
    # OTB Number of threads
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"]= str(nbcore)
        
    # =============================================================================
    #     The python module providing access to OTB applications is otbApplication
    # =============================================================================
    if OTBPythonBinding is True:
        
        try :
            import otbApplication
            
        except:
            OTBPythonBinding = False
            raise ImportError("You need to have OTB available in python to use OTBPythonBinding")
        
    else:        
        if resample20mBands:
            print("WARNING : Resample 20m bands at 10m is not available without OTB Python Binding")
            sys.exit(1)
        

    # =============================================================================
    #     Initiate variables and make dirs
    # =============================================================================
    
    outDir = os.path.dirname(outSITS)
    
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    
    band10m=['2', '3', '4', '8']
    band20m=['5', '6', '7','8A', '11', '12']
    
    if resample20mBands:
        bands = band10m+band20m
    else:
        bands = band10m
        
    
    # =============================================================================
    #     unzip only used bands
    # =============================================================================
    
    
    if unzip:
        print('unzipping used bands')            
        if resample20mBands:
            formula = "parallel -j "+str(nbcore)+" unzip -n {} *FRE_B?.tif *FRE_B??.tif *CLM_R1* *.xml *.jpg ::: "+os.path.join(S2Dir,"*.zip")
        else:
            formula = "parallel -j "+str(nbcore)+" unzip -n {} *FRE_B2*.tif *FRE_B3*.tif *FRE_B4*.tif *FRE_B8*.tif *CLM_R1* *.xml *.jpg ::: "+os.path.join(S2Dir,"*.zip")
        print('executing : '+formula)        
        os.system(formula)
        
    
    # =============================================================================
    #     Get Date for each acquisition and save to sample_time.csv
    # =============================================================================
        
    
    S2 = glob.glob(os.path.join(S2Dir,'SENTINEL2*/'))

    
    # =============================================================================
    #     If not Python Binding, must build vrt    
    # =============================================================================
    
    
    if not OTBPythonBinding:
        CLOUDS_temp = os.path.join(outDir+'clouds.tif')
        SITS_temp = os.path.join(outDir+'sits_temp.tif')

    import re
    regexYYYYMMDD = r"(?<!\d)(?:(?:20\d{2})(?:(?:(?:0[13578]|1[02])31)|(?:(?:0[1,3-9]|1[0-2])(?:29|30)))|(?:(?:20(?:0[48]|[2468][048]|[13579][26]))0229)|(?:20\d{2})(?:(?:0?[1-9])|(?:1[0-2]))(?:0?[1-9]|1\d|2[0-8]))(?!\d)"
    p = re.compile(regexYYYYMMDD)
    YYYYMMDDstart = p.search(S2[0]).start()
    YYYYMMDDend = p.search(S2[0]).end()
    
    sampleTime = [p.findall(S2folder)[0] for S2folder in S2]
    sampleTimeCsv = os.path.join(S2Dir,'sample_time.csv')
    np.savetxt(sampleTimeCsv,np.sort(np.asarray(sampleTime,dtype=np.int)),fmt='%d')
    
    
    # =============================================================================
    #     Order directory according to date
    # =============================================================================
    
    orderedSITS = sorted(S2, key=lambda x: x[YYYYMMDDstart:YYYYMMDDend])
    
    
    # =============================================================================
    #     Building cloud mask
    # =============================================================================
        
    cloudsToMask = [glob.glob(os.path.join(S2Date,'MASKS/*CLM_R1.tif'))[0] for S2Date in orderedSITS]    
    
    if OTBPythonBinding :
        appMask = otbApplication.Registry.CreateApplication("ConcatenateImages")
        appMask.GetParametersKeys()
        appMask.SetParameterStringList('il',cloudsToMask)
        appMask.Execute()
            
    else :
        # We print the keys of all its parameter
        print('building temporary cloud mask')    
        def listToStr(fileName,sep=' '):
            strList =''
            for file in fileName:
                strList = strList+sep+str(file)
            
            return strList
        
    
        os.system('gdalbuildvrt -separate {0}{1}'.format(CLOUDS_temp,listToStr(cloudsToMask)))
    
    
    # =============================================================================
    #     Building temporary SITS (with 4 bands at 10m, or 10 bands)
    # =============================================================================
            
    fourBandsToVrt = []
    
    if resample20mBands:
        sixBandsToVrt = []
        refBands = []
    for i in orderedSITS:
        for j in band10m:
            fourBandsToVrt.append(glob.glob(os.path.join(i,'*FRE_B{}.tif'.format(j)))[0])
        if resample20mBands:
            for k in band20m:
                sixBandsToVrt.append(glob.glob(os.path.join(i,'*FRE_B{}.tif'.format(k)))[0])
                refBands.append(glob.glob(os.path.join(i,'*FRE_B8.tif'))[0])
                
    if OTBPythonBinding :
        appTempSITS = otbApplication.Registry.CreateApplication("ConcatenateImages")
        
        appTempSITS.SetParameterStringList('il',fourBandsToVrt)
        
        appTempSITS.Execute()
        
        if resample20mBands:
            # Concatenate Band 8 as reference
            appReference = otbApplication.Registry.CreateApplication("ConcatenateImages")
            appReference.SetParameterStringList('il',refBands)
            appReference.Execute()
            # Concatenate 20m bands
            appToReproject= otbApplication.Registry.CreateApplication("ConcatenateImages")
            appToReproject.SetParameterStringList('il',sixBandsToVrt)
            appToReproject.Execute()
            
            # Resample 20m bands at 10m with Band 8
            appResampleCompute = otbApplication.Registry.CreateApplication("Superimpose")
            
            appResampleCompute.SetParameterInputImage('inr',appReference.GetParameterOutputImage("out"))
            appResampleCompute.SetParameterInputImage('inm',appToReproject.GetParameterOutputImage("out"))
            
            appResampleCompute.Execute()  
            
            # ===================================================================================
            #   Generate expression to reorder bands as b02,b03,b04,b08,b05,b06,b07,b08a,b11,b12
            # ===================================================================================
            
            reorderExp = ''
            for idx in range(len(orderedSITS)):
                start10m = idx*4+1
                start20m = idx*6+1
                
                im1b = ['im1b'+str(i) for i in range(start10m,start10m+4)]
                im2b = ['im2b'+str(i) for i in range(start20m,start20m+6)]
                currentExp = ';'.join(str(e) for e in im1b)+';'+';'.join(str(e) for e in im2b)+';'

                reorderExp += currentExp
            reorderExp = str(reorderExp[:-1])
            appConcatenatePerDate = otbApplication.Registry.CreateApplication("BandMathX")
            # first im inpput : temp SITS of the 4 10m bands for each date
            appConcatenatePerDate.AddImageToParameterInputImageList('il',appTempSITS.GetParameterOutputImage("out"))
            # second im inpput : resample 10m SITS of the 6 20m bands for each date
            appConcatenatePerDate.AddImageToParameterInputImageList('il',appResampleCompute.GetParameterOutputImage("out"))            
            # ordered as given in reorderExp
            appConcatenatePerDate.SetParameterString('exp',reorderExp)
            appConcatenatePerDate.Execute()
                
            
    else :            
        print('building temporary SITS with no gapfilling')    
        os.system('gdalbuildvrt -separate {0}{1}'.format(SITS_temp,listToStr(fourBandsToVrt)))
        
        
    # =============================================================================
    #     Execute process
    # =============================================================================
         
    
    print('Building SITS...')
    
    if OTBPythonBinding:
        app = otbApplication.Registry.CreateApplication("ImageTimeSeriesGapFilling")
        
        # We print the keys of all its parameter
        
        if resample20mBands : 
            app.SetParameterInputImage("in",appConcatenatePerDate.GetParameterOutputImage("out"))
        else:
            app.SetParameterInputImage("in",appTempSITS.GetParameterOutputImage("out"))
            
        app.SetParameterInputImage("mask",appMask.GetParameterOutputImage("out"))
        app.SetParameterString("out",outSITS)
        #app.SetParameterString("in",SITS_temp)
        #app.SetParameterString("mask",CLOUDS_temp)
        app.SetParameterOutputImagePixelType("out", otbApplication.ImagePixelType_int16) #int16 = 1
        app.SetParameterString("it",'linear')
        app.SetParameterString("id",sampleTimeCsv)
        app.SetParameterInt("comp",len(bands))
        app.SetParameterInt("ram",ram)
        if resampleCSV:
            app.SetParameterString("od",resampleCSV)    
        app.ExecuteAndWriteOutput()
        
    else:
        if resampleCSV :
            formula = "otbcli_ImageTimeSeriesGapFilling -in {0} -mask {1} \
                -out {2} int16 -comp {3} -it linear \
                -id {4} -od {5}".format(SITS_temp,CLOUDS_temp,outSITS,len(bands),sampleTimeCsv,resampleCSV)
        else:
            formula = "otbcli_ImageTimeSeriesGapFilling -in {0} -mask {1} \
                -out {2} int16 -comp {3} -it linear \
                -id {4} ".format(SITS_temp,CLOUDS_temp,outSITS,len(bands),sampleTimeCsv)
            
        print('Formula is '+formula)
    
        os.system(formula)
    
        os.remove(CLOUDS_temp)
        
        os.remove(SITS_temp)
    print('SITS built at '+str(outSITS))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        prog = os.path.basename(sys.argv[0])
        print(sys.argv[0]+' [options]')
        print("Help : ", prog, " --help")
        print("or : ", prog, " -h")
        print("example 1 : python %s -s2dir /tmp/S2downloads -out /tmp/SITS.tif"%sys.argv[0])
        print("example 2 : python %s -s2dir /tmp/S2downloads -out /tmp/SITS.tif -resample20m True -unzip True -nbcore 4 -ram 4000"%sys.argv[0])
        sys.exit(-1)  
 
    else:
        usage = "usage: %prog [options] "
        parser = argparse.ArgumentParser(description = "Compute Satellite Image Time Series from Sentinel-2 A/B.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        parser.add_argument("-s2dir","--wd", dest="s2dir", action="store",help="Sentinel-2 L2A Theia directory", required = True)
           
        parser.add_argument("-outSITS","--out", dest="outSITS", action="store", \
        help="Output name of the Sentinel-2 Image Time Series", required = True, type = str)
        
        parser.add_argument("-resample20m","--rs", dest="resample20mBands", action="store", \
        help="Resample the 20m bands at 10m for computing 10 bands per date", required = False, default = False)
        
        parser.add_argument("-resampleCSV","--rsCSV", dest="resampleCSV", action="store", \
        help="CSV of output dates", required = False, default = False)
        
        parser.add_argument("-unzip","--u", dest="unzip", action="store", \
        help="Do unzip of S2 images ?", required = False, type = bool, default = False)
        
        parser.add_argument("-OTBPythonBinding","--otb", dest="OTBPythonBinding", action="store", \
        help="If False, Compute VRT instead of using OTB Python Binding", required = False, default = True)
        
        parser.add_argument("-nbcore","--n", dest="nbcore", action="store", \
        help="Number of CPU / Threads to use for OTB applications (ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS)", \
        default = "4", required = False, type = int)                                   
        
        parser.add_argument("-ram","--r", dest="ram", action="store", \
        help="RAM for otb applications", default = "1", required = False, type = int)
        
        args = parser.parse_args()
    
        
        computeSITS(S2Dir=args.s2dir,outSITS=args.outSITS,resample20mBands=args.resample20mBands,\
                    resampleCSV=args.resampleCSV,unzip = args.unzip,OTBPythonBinding=args.OTBPythonBinding,nbcore=args.nbcore,ram=args.ram)
        
