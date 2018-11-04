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

import sys
import argparse
import os
import re
import numpy as np
import glob
import gdal

indicesDict = dict(
    NDVI='B04==-9999 ? -9999 : ndvi(B04,B08)*100',
    ACORVI='B04==-9999 ? -9999 : ndvi((B04+0.05),B08)*100',
    SAVI='(B08 + B04 + 0.5) == 0 ? -9999 : (1.5 * (B08 - B04) / (B08 + B04 + 0.5))*100',
    EVI='(B08 + 6*B04 - 7.5*B02 + 1) == 0 ? -9999 : 2.5*(B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)*100',
    EVI2='(B08 + 2.4 * B04 + 1) == 0 ? -9999 : 2.5 * (B08 - B04) / (B08 + 2.4 * B04 + 1)*100',
    PSRI='B05 == 0 ? -9999 : ((B04 - B02) / B05)*100',
    ARI2='B02 == 0 ? -9999 : B03 == 0 ? -9999 : ((B08 / B02) - (B08 / B03))')


def generateIndicesFromSITS(
        inSITS,
        outIndice,
        indice='ACORVI',
        sampleTime=False,
        comp=4,
        nbcore=1,
        ram=256):
    """
    Generate indices from Satellite Images Time Series.

    Parameters
    ----------
    inSITS : str
        Path where your SITS is located.
    outIndice : Output raster (vrt)
        Output name of your raster.
        Save as vrt format, int16 (indice multiplied by 100)
        Each date will be saved in a subfolder with indice name)
    indice : str, default 'ACORVI'
        Name of the indice to produce. List : NDVI,ACORVI,EVI,SAVI,EVI,EVI2,PSRI,ARI2
    sampleTime : str, default False
        CSV of the SITS sample time
    comp : int, default 4
        Number of components per date.
    nbcore : int, default 1
        Number of cores used.
    ram : int, default 256
        Available ram in mb.
    """
    # OTB Number of threads
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(nbcore)

    try:
        import otbApplication
    except BaseException:
        raise ImportError(
            "You need to have OTB available in python to use this script")

    # =============================================================================
    #     initialize variables
    # =============================================================================

    indiceFormula = indicesDict[indice]
    indiceFolder = glob.os.path.join(glob.os.path.dirname(outIndice), indice)

    if not glob.os.path.exists(indiceFolder):
        glob.os.makedirs(indiceFolder)

    bandOrder4 = np.array(['2', '3', '4', '8'])
    bandOrder10 = np.array(
        ['2', '3', '4', '8', '5', '6', '7', '8A', '11', '12'])

    if comp == 4:
        bandOrder = bandOrder4
    elif comp == 10:
        bandOrder = bandOrder10
    else:
        raise Exception(
            'This script only works with 4 or 10 composant per date')

    # =============================================================================
    #     compute number of dates
    # =============================================================================
    data_src = gdal.Open(inSITS)
    nBands = data_src.RasterCount
    nDates = int(nBands / comp)

    if sampleTime:
        TIMESAMPLE = np.loadtxt(sampleTime, dtype=np.int)

    # =============================================================================
    #     compute indice for each date
    # =============================================================================
    for i in range(1, nDates + 1):
        print('=' * 50)
        print('Computing date {}/{}'.format(i, nDates))

        indiceBandMathExp = indiceFormula
        if not glob.os.path.exists(indiceFolder):
            glob.os.makedirs(indiceFolder)
        if sampleTime:
            currentIndice = indiceFolder + '/' + indice + \
                '_' + str(TIMESAMPLE[i - 1]) + '.tif'
        else:
            currentIndice = indiceFolder + '/' + indice + '_' + str(i) + '.tif'

        bandsToChange = re.findall('B[0-9]*', indiceFormula)

        # =========================================================================
        #         replace formula with BandMath expression
        # =========================================================================
        for band in set(bandsToChange):
            originalBand = band[1:]
            bandIdx = np.where(bandOrder == str(int(originalBand)))[0][0]
            newBand = comp * (i - 1) + (bandIdx + 1)

            indiceBandMathExp = indiceBandMathExp.replace(
                band, 'im1b' + str(int(newBand)))

        # =========================================================================
        #         Execute BandMath
        # =========================================================================
        computeIndice = otbApplication.Registry.CreateApplication("BandMath")
        computeIndice.SetParameterStringList('il', [inSITS])
        computeIndice.SetParameterString('out', currentIndice)
        computeIndice.SetParameterString('exp', indiceBandMathExp)
        computeIndice.SetParameterInt("ram", ram)
        computeIndice.ExecuteAndWriteOutput()

        # replace -9999 by nodata
        glob.os.system('gdal_edit.py {0} -a_nodata -9999'.format(outIndice))

    # =============================================================================
    #     create list of each produced date and build vrt
    # =============================================================================

    def listToStr(fileName, sep=' '):
        strList = ''
        for file in fileName:
            strList = strList + sep + str(file)

        return strList

    indiceFiles = sorted(
        glob.glob(
            glob.os.path.join(
                indiceFolder,
                indice,
                '*.tif')))
    indiceFilesToBash = listToStr(indiceFiles)
    buildVrt = 'gdalbuildvrt -separate {} {}'.format(
        outIndice, indiceFilesToBash)
    glob.os.system(buildVrt)

    print(indice + ' has been calculated.')


if __name__ == "__main__":
    if len(sys.argv) == 1:
        prog = os.path.basename(sys.argv[0])
        print(sys.argv[0] + ' [options]')
        print("Help : ", prog, " --help")
        print("or : ", prog, " -h")
        print(
            "example 1 : python %s -inSITS /tmp/SITS.tif -outIndice /tmp/indice/ACORVI.vrt -indice ACORVI" %
            sys.argv[0])

        sys.exit(-1)

    else:

        usage = "usage: %prog [options] "
        parser = argparse.ArgumentParser(
            description="Compute Satellite Image Time Series from Sentinel-2 A/B.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument("-inSITS", "--in", dest="inSITS", action="store",
                            help="Path of your SITS", required=True)

        parser.add_argument(
            "-outIndice",
            "--out",
            dest="outIndice",
            action="store",
            help="Output name (vrt) of the indice (each date will be saved as a tif in subfolder)",
            required=True,
            type=str)

        parser.add_argument(
            "-indice",
            "--indice",
            dest="indice",
            action="store",
            help="Indice to generate",
            choices=list(
                indicesDict.keys()),
            default='ACORVI')

        parser.add_argument(
            "-sampleTime",
            "--s",
            dest="sampleTime",
            action="store",
            help="CSV of sample time (format YYYYMMDD)",
            required=False,
            default=False)

        parser.add_argument("-comp", "--c", dest="comp", action="store",
                            help="Number of components per date (4 or 10)",
                            required=False, type=int, default=4)

        parser.add_argument(
            "-nbcore",
            "--n",
            dest="nbcore",
            action="store",
            help="Number of CPU / Threads to use for OTB applications (ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS)",
            default="1",
            required=False,
            type=int)

        parser.add_argument("-ram", "--r", dest="ram", action="store",
                            help="RAM for otb applications",
                            default="256", required=False, type=int)

        args = parser.parse_args()

        generateIndicesFromSITS(
            inSITS=args.inSITS,
            outIndice=args.outIndice,
            indice=args.indice,
            sampleTime=args.sampleTime,
            comp=args.comp,
            nbcore=args.nbcore,
            ram=args.ram)
