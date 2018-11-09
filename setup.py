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
import setuptools
__version__ = 0.9

with open("README.md", "r") as fh:
    long_description = fh.read()

"""
install_requires=[
    'scikit-learn',
    'numpy',
    'scipy',
    'osgeo'],
"""
setuptools.setup(
    name='MuseoToolBox',
    version=str(__version__),
    description='Raster and vector tools for Remote Sensing and Classification',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/lennepkade/MuseoToolBox',
    author='Nicolas Karasiak',
    author_email='karasiak.nicolas@gmail.com',
    license='GPLv3',
    packages=setuptools.find_packages(),
    zip_safe=False,
    entry_points = {
        'console_scripts': [
            'mtb_sampleExtraction=MuseoToolBox.vectorTools.sampleExtraction:main',
        ],
    }
 )
