# coding: utf-8
from .. import raster_tools
import os
from scipy import stats
import numpy as np
import sys


class modalClass:
    """
    Compute modal class and number of agreements.

    Parameters
    -----------

    inRaster : str
        The raster (at least 2 bands)
    outRaster : str
        The output raster.
    inMaskRaster : str or False,
        The mask raster (0 = to mask, >0 : no mask).
    """

    def __init__(
            self,
            inRaster,
            outRaster,
            inMaskRaster):
        #process = raster_tools.readAndWriteRaster(inRaster,outRaster=outRaster,inMaskRaster=inMaskRaster,outNBand=2,outGdalGDT=outGdalDT,outNoData=outNoData)

        process = raster_tools.rasterMath(inRaster, inMaskRaster)
        process.addFunction(self.stabCalc, outRaster, 2, 3, 0)
        process.run()

    def stabCalc(self, arr):
        """
        Compute modal and number of agreements.

        Parameters
        -----------
        arr : array.
            The array where to compute stats from.

        Returns
        --------
        arr : array of shape (:,2).
        """
        tmp = stats.mode(arr, axis=-1)
        tmpStack = np.column_stack((tmp[0], tmp[1]))
        return tmpStack


def __modalClassCLI(argv=None, apply_config=True):
    import argparse
    if len(sys.argv) == 1:
        prog = os.path.basename(sys.argv[0])
        print(sys.argv[0] + ' [options]')
        print("Help : ", prog, " --help")
        print("or : ", prog, " -h")
        print(
            2 *
            ' ' +
            "example 1 : ",
            prog,
            " -in raster.tif -out modal.tif")
        sys.exit(-1)

    else:
        usage = "usage: %prog [options] "
        parser = argparse.ArgumentParser(
            description="Compute modal class (first band) and number of agreements (second band).",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "-in",
            "--image",
            dest="inRaster",
            action="store",
            help="Image to extract values from and to generate centroid from output vector",
            required=True)

        parser.add_argument(
            "-inm",
            "--inMaskRaster",
            dest="inMaskRaster",
            action="store",
            help="Vector to fill with raster values",
            required=False,
            default=False,
            type=str)

        parser.add_argument(
            "-out",
            "--outRaster",
            dest="outRaster",
            action="store",
            help="Raster to save (geotif)",
            required=True,
            type=str)
        args = parser.parse_args()

        modalClass(
            inRaster=args.inRaster,
            outRaster=args.outRaster,
            inMaskRaster=args.inMaskRaster)


if __name__ == "__main__":
    sys.exit(__modalClassCLI())
