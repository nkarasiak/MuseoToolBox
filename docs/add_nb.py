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

import glob
path = '../docs/source/notebooks/**/*.ipynb'
ipynbs = glob.glob(path)

rstNb = open("source/notebooks.rst","w+")
rstNb.write('Notebooks\n==================\n' )
folder=[]
for ipynb in ipynbs:
    fld = ipynb.split('/')[4]
    if not fld in folder:
        folder.append(fld)
        rstNb.write('\n'+fld+'\n')
        rstNb.write((len(fld)*1+2)*'-'+'\n')
        rstNb.write('.. toctree::\n')
    rstNb.write('    notebooks/'+fld+'/'+glob.os.path.basename(ipynb)+'\n')
        
rstNb.close()