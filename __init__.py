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

from __future__ import absolute_import 

from .metadata.version import version as __version__
# import rasterTools, vectorTools and other directly in root : mtb.XXX
#from .tools import *
from . import vectorTools,rasterTools

# import others folder as mtb.Folder.XXX
from . import Sentinel2

# Maybe to activate later ?
# You can import via :
# from MuseoToolBox import apps as mtbApps
#from . import stats 
#from . import apps


