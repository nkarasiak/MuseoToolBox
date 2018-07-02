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

import numpy as np
import glob
import os
import sys,argparse

def resampleEveryXDays(S2Dir,out,nDays=5,startDate=False,lastDate=False):
    """
    Generate a custom Sample Time for Satellite Image Time Series.
    
    Parameters
    ----------
    S2Dir : Directory
        Directory where unzip files from THEIA L2A are unzipped, or zipped.
    out : str (csv type).
        Output csv with new samle time.
    nDays : int, default 5
        Integer, days delta to generate.
    startDate : int, default False.
        If specified, format (YYYYMMDD).
    lastDate : int, default False.
        If specified, format (YYYYMMDD).
    """
    try:
        import datetime as dt
    except:
        raise ImportError('Datetime python library is needed to compute custom sample time')

    
    # =============================================================================
    #     List all subfolders which begins with SENTINEL2
    #     if no lastDate and startDate given
    # =============================================================================

    if lastDate is False or startDate is False: 
        S2 = glob.glob(glob.os.path.join(S2Dir,'SENTINEL2*/'))
        
        # if no folder, looking for zip files
        if S2 == [] :
            S2 =  glob.glob(glob.os.path.join(S2Dir,'SENTINEL2*.zip'))
            
        else:
            S2 = [os.path.basename(os.path.dirname(S2Folder)) for S2Folder in S2]
        
        # ==========================================================================
        #     Detecting YYYYMMDD date format
        # ==========================================================================
        
        import re
        regexYYYYMMDD = r"(?<!\d)(?:(?:20\d{2})(?:(?:(?:0[13578]|1[02])31)|(?:(?:0[1,3-9]|1[0-2])(?:29|30)))|(?:(?:20(?:0[48]|[2468][048]|[13579][26]))0229)|(?:20\d{2})(?:(?:0?[1-9])|(?:1[0-2]))(?:0?[1-9]|1\d|2[0-8]))(?!\d)"
        p = re.compile(regexYYYYMMDD)
        
        sampleTime = sorted([p.findall(S2folder)[0] for S2folder in S2])
        
        sampleTimeDT = [dt.datetime.strptime(date,'%Y%m%d') for date in sampleTime]
    
    # =============================================================================
    #     If startDate use it and convert it to datetime format    
    #     Else, use lastDate found in folder
    # =============================================================================
    if startDate is False : startDate = sampleTimeDT[0]
    else : startDate = dt.datetime.strptime(startDate,'%Y%m%d')
    
    # =============================================================================
    #     If lastDate use it and convert it to datetime format
    #     Else, use lastDate found in folder
    # =============================================================================
    if lastDate is False : lastDate = sampleTimeDT[-1]
    else : lastDate = dt.datetime.strptime(lastDate,'%Y%m%d')
    
    customeSampleTime = [startDate.strftime('%Y%m%d')]
    newDate = startDate
    while newDate < lastDate : 
        newDate = newDate+dt.timedelta(nDays)
        customeSampleTime.append(newDate.strftime('%Y%m%d'))
    
    np.savetxt(out,np.asarray(customeSampleTime,dtype=np.int),fmt='%d')
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        prog = os.path.basename(sys.argv[0])
        print(sys.argv[0]+' [options]')
        print("Help : ", prog, " --help")
        print("or : ", prog, " -h")
        print("example 1 : python %s -s2dir /tmp/S2downloads -out /tmp/sample_time.csv"%sys.argv[0])
        print("example 2 : python %s -s2dir /tmp/S2downloads -out /tmp/sample_time.csv -startDate 20170103 -endDate 20171225"%sys.argv[0])
        sys.exit(-1)  
 
    else:
        usage = "usage: %prog [options] "
        parser = argparse.ArgumentParser(description = "Compute Satellite Image Time Series from Sentinel-2 A/B.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        parser.add_argument("-s2dir","--wd", dest="s2dir", action="store",help="Sentinel-2 L2A Theia directory (if images not unzipped, will search for zip)",\
        required = False)
           
        parser.add_argument("-out","--o", dest="out", action="store", \
        help="Csv file", required = True, type = str)
        
        parser.add_argument("-nDays","--n", dest="nDays", action="store", \
        help="New delta day to generate", required = True, type = int)
        
        parser.add_argument("-startDate","--s", dest="startDate", action="store", \
        help="If specified, format (YYYYMMDD).", required = False, default = False)
        
        parser.add_argument("-lastDate","--l", dest="lastDate", action="store", \
        help="If specified, format (YYYYMMDD).", required = False, default = False)
        
        args = parser.parse_args()
    
        resampleEveryXDays(S2Dir = args.s2dir,out=args.out,nDays=args.nDays,startDate=args.startDate,lastDate=args.lastDate)
        
