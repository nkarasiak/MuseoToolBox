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

#import multiprocessing
import gdal
import numpy as np
import os

def convertGdalDataTypeToOTB(gdalDT):
    """
    Convert Gdal DataType to OTB str format.
    
    Parameters
    ----------
    gdalDT : int
        gdal Datatype (integer value)
    
    Output
    ----------
    str format of OTB datatype
        availableCode = uint8/uint16/int16/uint32/int32/float/double
    """
    code = ['uint8','uint16','int16','uint32','int32','float','double']
    
    return code[gdalDT]


def get_coords_from_roi(raster_name,roi_name):
    raster = gdal.Open(raster_name,gdal.GA_ReadOnly)
    if raster is None:
        print('Impossible to open '+raster_name)
        #exit()

    ## Open ROI
    roi = gdal.Open(roi_name,gdal.GA_ReadOnly)
    if roi is None:
        print('Impossible to open '+roi_name)
        #exit()

    if stand_name:
        ## Open Stand
        stand = gdal.Open(stand_name,gdal.GA_ReadOnly)
        if stand is None:
            print('Impossible to open '+stand_name)
            #exit()

    ## Some tests
    if (raster.RasterXSize != roi.RasterXSize) or (raster.RasterYSize != roi.RasterYSize):
        print('Images should be of the same size')
        #exit()

def get_samples_from_roi(raster_name,roi_name,stand_name=False,getCoords=False,onlyCoords=False):
    '''!@brief Get the set of pixels given the thematic map.
    Get the set of pixels given the thematic map. Both map should be of same size. Data is read per block.
        Input:
            raster_name: the name of the raster file, could be any file that GDAL can open
            roi_name: the name of the thematic image: each pixel whose values is greater than 0 is returned
        Output:
            X: the sample matrix. A nXd matrix, where n is the number of referenced pixels and d is the number of variables. Each 
                line of the matrix is a pixel.
            Y: the label of the pixel
    Written by Mathieu Fauvel.
    ''' 
    ## Open Raster
    raster = gdal.Open(raster_name,gdal.GA_ReadOnly)
    if raster is None:
        print('Impossible to open '+raster_name)
        #exit()

    ## Open ROI
    roi = gdal.Open(roi_name,gdal.GA_ReadOnly)
    if roi is None:
        print('Impossible to open '+roi_name)
        #exit()

    if stand_name:
        ## Open Stand
        stand = gdal.Open(stand_name,gdal.GA_ReadOnly)
        if stand is None:
            print('Impossible to open '+stand_name)
            #exit()

    ## Some tests
    if (raster.RasterXSize != roi.RasterXSize) or (raster.RasterYSize != roi.RasterYSize):
        print('Images should be of the same size')
        #exit()

    ## Get block size
    band = raster.GetRasterBand(1)
    block_sizes = band.GetBlockSize()
    x_block_size = block_sizes[0]
    y_block_size = block_sizes[1]
    del band
    
    ## Get the number of variables and the size of the images
    d  = raster.RasterCount
    nc = raster.RasterXSize
    nl = raster.RasterYSize
    
    ulx, xres, xskew, uly, yskew, yres  = roi.GetGeoTransform()
    
    if getCoords :
        coords = np.array([],dtype=np.uint16).reshape(0,2)

    ## Read block data
    X = np.array([]).reshape(0,d)
    Y = np.array([]).reshape(0,1)
    STD = np.array([]).reshape(0,1)
    
    total = nl*y_block_size
    for i in range(0,nl,y_block_size):

        if i + y_block_size < nl: # Check for size consistency in Y
            lines = y_block_size
        else:
            lines = nl - i
        for j in range(0,nc,x_block_size): # Check for size consistency in X
            if j + x_block_size < nc:
                cols = x_block_size
            else:
                cols = nc - j

            # Load the reference data
            
            ROI = roi.GetRasterBand(1).ReadAsArray(j, i, cols, lines)
            if not onlyCoords:
                if stand_name:
                    STAND = stand.GetRasterBand(1).ReadAsArray(j, i, cols, lines)
            
            t = np.nonzero(ROI)
            
            if t[0].size > 0:
                Y = np.concatenate((Y,ROI[t].reshape((t[0].shape[0],1)).astype('uint8')))
                if stand_name:
                    STD = np.concatenate((STD,STAND[t].reshape((t[0].shape[0],1)).astype('uint8')))
                if getCoords :
                    #coords = sp.append(coords,(i,j))
                    #coordsTp = sp.array(([[cols,lines]]))
                    #coords = sp.concatenate((coords,coordsTp))
                    #print(t[1])
                    #print(i)
                    #sp.array([[t[1],i]])
                    coordsTp = np.empty((t[0].shape[0],2))
                    coordsTp[:,0] = t[1]
                    coordsTp[:,1] = [i]*t[1].shape[0]
                    """
                    for n,p in enumerate(coordsTp):
                        coordsTp[n] = pixel2coord(p)
                    """
                    coords = np.concatenate((coords,coordsTp))

                # Load the Variables
                if not onlyCoords:
                    Xtp = np.empty((t[0].shape[0],d))
                    for k in range(d):
                        band = raster.GetRasterBand(k+1).ReadAsArray(j, i, cols, lines)
                        Xtp[:,k] = band[t]
                    try:
                        X = np.concatenate((X,Xtp))
                    except MemoryError:
                        print('Impossible to allocate memory: ROI too big')
                        exit()
    
    
    """
    # No conversion anymore as it computes pixel distance and not metrics
    if convertTo4326:
        import osr
        from pyproj import Proj,transform
        # convert points coords to 4326
        # if vector 
        ## inShapeOp = ogr.Open(inVector)
        ## inShapeLyr = inShapeOp.GetLayer()
        ## initProj = Proj(inShapeLyr.GetSpatialRef().ExportToProj4()) # proj to Proj4
        
        sr = osr.SpatialReference()
        sr.ImportFromWkt(roi.GetProjection())
        initProj = Proj(sr.ExportToProj4())
        destProj = Proj("+proj=longlat +datum=WGS84 +no_defs") # http://epsg.io/4326
        
        coords[:,0],coords[:,1] = transform(initProj,destProj,coords[:,0],coords[:,1]) 
    """
    
    # Clean/Close variables
    # del Xtp,band    
    roi = None # Close the roi file
    raster = None # Close the raster file
    
    if onlyCoords:
        return coords
    if stand_name:
        if not getCoords:
            return X,Y,STD
        else:
            return X,Y,STD,coords
    elif getCoords :
        return X,Y,coords
    else:
        return X,Y

def rasterize(data,vectorSrc,field,outFile):
    dataSrc = gdal.Open(data)
    import ogr
    shp = ogr.Open(vectorSrc)

    lyr = shp.GetLayer()

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(outFile,dataSrc.RasterXSize,dataSrc.RasterYSize,1,gdal.GDT_Byte)
    dst_ds.SetGeoTransform(dataSrc.GetGeoTransform())
    dst_ds.SetProjection(dataSrc.GetProjection())
    if field is None:
        gdal.RasterizeLayer(dst_ds, [1], lyr, None)
    else:
        OPTIONS = ['ATTRIBUTE='+field]
        gdal.RasterizeLayer(dst_ds, [1], lyr, None,options=OPTIONS)
    
    data,dst_ds,shp,lyr=None,None,None,None
    return outFile

def pushFeedback(message,feedback=None):
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


  
class readAndWriteRaster:
    """
    Read a raster per block, and perform one or many functions to one or many raster outputs.
    
    - If you want to add an output, just use addOutput() function and select raster path, number of bands and datatype.
    - If you want a sample of your data, just call getRandomBlock().
    
    Parameters
    ----------
    inRaster : str
        Gdal supported raster
    outRaster : str
        Path to raster to save (tif file)
    inMaskRaster : str, default False.
        Path to mask raster (masked values are 0). If not, please change self.maskNoData to your value.
    outNBand : int, default 1
        Number of bands of the first output.
    outGdalGDT : int, default 1.
        Gdal DataType of the first output.
    outNoData : int, default False.
        If False, will set the same value as the input raster nodata.
    parallel : int, default False.
        Number of cores to be used if not False.
        
    Output
    ----------
        As many raster (geoTiff) as output defined by the user.
    
    """   
    
    def __init__(self,inRaster,inMaskRaster=False,parallel=False):
        self.parallel = parallel
        
        #if parallel is not False:
        
            
        
        self.openRaster = gdal.Open(inRaster,gdal.GA_ReadOnly)
        if self.openRaster is None:
            # fix_print_with_import
            print('Impossible to open '+inRaster)
            exit()
            
        self.nb  = self.openRaster.RasterCount
        self.nc = self.openRaster.RasterXSize
        self.nl = self.openRaster.RasterYSize
        
        
        # Get the geoinformation
        self.GeoTransform = self.openRaster.GetGeoTransform()
        self.Projection = self.openRaster.GetProjection()
    
        # Get block size
        band = self.openRaster.GetRasterBand(1)
        block_sizes = band.GetBlockSize()
        self.x_block_size = block_sizes[0]
        self.y_block_size = block_sizes[1]
        self.total = self.nl #/self.y_block_size

        self.nodata = band.GetNoDataValue()
        
        if self.nodata is None:
            self.nodata = -9999
        
        del band
    
        #
        self.mask = inMaskRaster
        if self.mask:
            self.maskNoData = 0
            self.openMask = gdal.Open(inMaskRaster)
        ## Initialize the output
        self.functions = []        
        self.outputs = []
        #out = dst_ds.GetRasterBand(1)
        self.lastProgress = 0
        self.outputNoData = []
        
    def addFunction(self,function,outRaster,outNBand,outGdalGDT=11,outNoData=False):
        self.driver = gdal.GetDriverByName('GTiff')
        self.__addOutput__(outRaster,outNBand,outGdalGDT)
        self.functions.append(function)        
        self.outputNoData.append(outNoData)

    def __addOutput__(self,outRaster,outNBand,outGdalGDT):
        if not os.path.exists(os.path.dirname(outRaster)):
            os.makedirs(os.path.dirname(outRaster))
        dst_ds = self.driver.Create(outRaster, self.nc,self.nl, outNBand, outGdalGDT)
        dst_ds.SetGeoTransform(self.GeoTransform)
        dst_ds.SetProjection(self.Projection)
        self.outputs.append(dst_ds)
        
    def __iterBlock__(self,getBlock=False):       
        for row in range(0, self.nl, self.y_block_size):
            for col in range(0, self.nc, self.x_block_size):
                width = min(self.nc - col, self.x_block_size)
                height = min(self.nl - row, self.y_block_size)
                
                if getBlock :
                    X,mask = self.generateBlockArray(col,row,width,height,self.mask)
                    yield X,mask,col,row,width,height
                else:
                    yield col,row,width,height
    
    def generateBlockArray(self,col,row,width,height,mask=True):       
        arr = np.empty((height*width,self.nb))
        
        for ind in range(self.nb):
            band = self.openRaster.GetRasterBand(int(ind+1))
            arr[:,ind] = band.ReadAsArray(col,row,width,height).reshape(width*height)
        if mask:
            arrMask = self.openMask.GetRasterBand(1).ReadAsArray(col,row,width,height).reshape(width*height)
        else:
            arrMask=None
            
        arr,arrMask = self.filterNoData(arr,arrMask)    
        
        return arr,arrMask
    
    def filterNoData(self,arr,mask=None):
        outArr = np.zeros((arr.shape))
        outArr[:] = self.nodata
        
        if self.mask :
            t = np.logical_or((mask==self.maskNoData),arr[:,0]==self.nodata)            
        else:
            t = np.where(arr[:,0]==self.nodata)
            
        tmpMask = np.ones(arr.shape,dtype=bool)            
        tmpMask[t,:] = False
        outArr[tmpMask] = arr[tmpMask]

        return outArr,tmpMask
    
    def getRandomBlock(self):    
        cols = int(np.random.permutation(range(0,self.nl,self.y_block_size))[0])
        lines = int(np.random.permutation(range(0,self.nc,self.x_block_size))[0])
        width = min(self.nc - lines, self.x_block_size)
        height = min(self.nl - cols, self.y_block_size)
    
        tmp,mask = self.generateBlockArray(lines,cols,width,height,self.mask)
        arr = tmp[mask[:,0],:]
        return arr
    


    def progressBar(self,line,total,length=40):
        
        try:
            self.percent
        except:
            self.percent = int((line+1)/(total)*100)
        
        if self.percent != int((line+1)/(total)*100):
            self.percent = int((line+1)/(total)*100)
            hashtag = int(self.percent/(100/length))
            empty = length-1-hashtag
            print(str('\r ['+hashtag*'#'+empty*' '+'] ')+str(self.percent)+'%',end='')
        
    def run(self,verbose=1,qgsFeedback=False):
        """
        Process with outside function.
        """            
        if self.parallel :
            try:
                from joblib import Parallel, delayed  
            except :
                raise ImportError('Please install joblib to use multiprocessing')
            def processParallel(X,mask,i,j,cols,lines,outputNBand,fun):                
                tmp = np.copy(X)
                tmp[mask[:,0],:outputNBand] = fun(tmp[mask[:,0],:])
                
                return tmp,mask,i,j,cols,lines,idx
                
                #return X
            #self.resFromIterBlock= []
            #self.resFromIterBlock.extend(self.__iterBlock__(getBlock=False))
            
            for idx,fun in enumerate(self.functions):
                outputNBand = self.outputs[idx].RasterCount 
                for X,mask,i,j,cols,lines,idx in Parallel(n_jobs=self.parallel)(delayed(processParallel)(X,mask,i,j,cols,lines,outputNBand,fun) for X,mask,i,j,cols,lines in self.__iterBlock__(getBlock=True)):
                    if verbose:
                        self.progressBar(j,self.total)
                    #for X,mask,i,j,cols,lines,idx in Parallel(n_jobs=self.parallel,verbose=False)(delayed(processParallel)(X,mask,i,j,cols,lines,outputNBand,fun) for X,mask,i,j,cols,lines in self.__iterBlock__(getBlock=True)):
                    for ind in range(self.outputs[idx].RasterCount):
                        indGdal = int(ind+1)
                        curBand = self.outputs[idx].GetRasterBand(indGdal)
                        curBand.WriteArray(X[:,ind].reshape(lines,cols),i,j)
                        curBand.FlushCache()
                    self.outputs[idx].GetRasterBand(1).SetNoDataValue(self.outNoData)

        else:
        
            for X,mask,col,line,cols,lines in self.__iterBlock__(getBlock=True):
                X_ = np.copy(X)
                actualProgress = int(line/self.total*100)                
                
                if self.lastProgress != actualProgress:
                    self.lastProgress = actualProgress 
    
                    if qgsFeedback:
                        if qgsFeedback == 'gui':                        
                            self.progressBar(line,self.total)
                        else:
                            qgsFeedback.setProgress(actualProgress)
                            if qgsFeedback.isCanceled():
                                break
                        
                    if verbose:
                        self.progressBar(line,self.total)
                    
                for idx,fun in enumerate(self.functions):
                    if self.outputNoData[idx] == False:
                        self.outputNoData[idx] = self.nodata
                        
                    maxBands = self.outputs[idx].RasterCount
                    if X_[mask].size > 0:
                        resFun = fun(X_[mask[:,0],:])
                        if maxBands > self.nb:
                            X = np.zeros((X_.shape[0],maxBands))
                            X[:,:] = self.outputNoData[idx]
                            X[mask[:,0],:] = resFun
                        if resFun.ndim == 1:
                            resFun = resFun.reshape(-1,1)
                        if resFun.shape[1] > maxBands :
                            raise ValueError ("Your function output {} bands, but your output is specified to have a maximum of {} bands.".format(resFun.shape[1],maxBands))
                        else:
                            X[:,:] = self.outputNoData[idx]
                            X[mask[:,0],:maxBands] = resFun
                    for ind in range(self.outputs[idx].RasterCount):
                        indGdal = int(ind+1)
                        curBand = self.outputs[idx].GetRasterBand(indGdal)
                        curBand.WriteArray(X[:,ind].reshape(lines,cols),col,line)
                        curBand.FlushCache()
                    self.outputs[idx].GetRasterBand(1).SetNoDataValue(self.outputNoData[idx])
                    
    
