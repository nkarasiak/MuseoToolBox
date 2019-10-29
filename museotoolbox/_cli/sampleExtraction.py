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

from ..raster_tools import sampleExtraction

import sys
import os
import numpy as np


def main(argv=None, apply_config=True):
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
            " -in raster.tif -vec roi.sqlite -out vector.sqlite -outfield.prefix.name band_ ")
        print(
            2 *
            ' ' +
            "example 2 : ",
            prog,
            " -in raster.tif -vec roi.sqlite -out vector.sqlite -field ogc_fid")
        sys.exit(-1)

    else:
        usage = "usage: %prog [options] "
        parser = argparse.ArgumentParser(
            description="From points or polygons, extraction each pixel centroid and extract values from raster.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            "-in",
            "--image",
            dest="inRaster",
            action="store",
            help="Image to extract values from and to generate centroid from output vector",
            required=True)

        parser.add_argument(
            "-vec",
            "--vector",
            dest="inVector",
            action="store",
            help="Vector to fill with raster values",
            required=True,
            type=str)

        parser.add_argument(
            "-out",
            "--outvector",
            dest="outVector",
            action="store",
            help="Vector to save (sqlite or gpkg extension if possible)",
            required=True,
            type=str)

        parser.add_argument(
            "-field",
            "--uniqueField",
            dest="uniqueFID",
            action="store",
            help="Unique field per feature. If no field, will create an 'uniquefid' field in the original shapefile.",
            required=False,
            default=None)

        parser.add_argument(
            "-outfield.prefix.name",
            "--outField",
            dest="bandPrefix",
            action="store",
            help="Prefix name to save the values from the raster. E.g 'band_'.",
            required=False,
            default=None)

        args = parser.parse_args()

        sampleExtraction(
            inRaster=args.inRaster,
            inVector=args.inVector,
            outVector=args.outVector,
            uniqueFID=args.uniqueFID,
            bandPrefix=args.bandPrefix)


if __name__ == "__main__":
    sys.exit(main())
