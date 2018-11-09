# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import function_dataraster as funraster
import os
from scipy import stats

YEAR = [2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
CLASSIFIER = ['svm', 'gmm']
arr = np.array([[5, 5, 6], [6, 6, 6], [5, 5, 5], [5, 6, 7]])
# WDIR='/media/nkarasiak/DATA/Formosat_2006-2014/Maps/'
WDIR = '/mnt/DATA/Formosat_2006-2014/Maps'

W, H, D = 4531, 4036, 9

for classifier in CLASSIFIER:
    os.chdir(WDIR)
    # Initializa output
    imc = sp.empty((H, W, D), dtype='uint8')
    out = sp.zeros((H, W, 2), dtype='uint8')
    # Load data
    for i, year in enumerate(YEAR):
        imc[:, :, i], GeoTransform, Projection = funraster.open_data(
            'map_' + classifier + '_' + str(year) + '.tif')

    # Compute the most frequent class
    t = sp.where(imc[:, :, 0] > 0)
    tx, ty = t[0], t[1]
    for tx, ty in zip(t[0], t[1]):
        tempMode = stats.mode(imc[tx, ty, :], axis=None)
        # Class and number of mode
        out[tx, ty, 0], out[tx, ty, 1] = tempMode[0], tempMode[1]

    # Save the data
    funraster.write_data(
        'count_' +
        classifier +
        '.tif',
        out,
        GeoTransform,
        Projection)
    # os.system('gdal_translate -a_nodata 0 -projwin 541989.189387 6262294.656 547522.004771 6258905.18352 -of GTiff -ot Byte temp.tif temp_c.tif')
    # os.system('otbcli_ColorMapping -in temp_c.tif -out count_'+classifier+'.png uint8 -method custom -method.custom.lut lut_s.txt')
    # os.system('rm temp*.tif')
    imc, out = [], []
    print('Finished count for file "count_' + classifier + '.tif"')
    del tempMode


qml = '/home/nkarasiak/GDrive/TEFOR/Data/couleurs_nomenclature.qml'


def loadQML(qml):

    a = open(qml)
    valueList = []
    labelList = []
    colorList = []
    #customQML = []
    #cdict = {}
    for i in a:
        if i.rfind('value="') != -1:
            value = i[i.rfind('value="') + 7:i.rfind('label="') - 2]
            valueList.append(int(value))
            label = i[i.rfind('label="') + 7:i.rfind('color="') - 2]
            labelList.append(label)
            color = i[i.rfind('color="') + 7:i.rfind("/'") - 3]
            # colorList.append(colors.hex2color(color))
            colorList.append(color)
            #customQML.append((int(value), colors.hex2color(color)))
    return valueList, labelList, colorList


level3values, speciesName, speciesColors = loadQML(qml)


nomenclature = sp.loadtxt(
    '/home/nkarasiak/GDrive/TEFOR/Code/S2/nomenclature.csv',
    delimiter=',',
    dtype=str)

xx = out[sp.where(out[:, :, 0] != 0)]
boxprops = dict(linestyle='--', linewidth=1, color='black')
flierprops = dict(marker='o', markerfacecolor='green', markersize=12,
                  linestyle='none')
medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick')
meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')

labels = []
agreements = []
for label in sp.unique(xx):
    labels.append(label)
    y = xx[:, 1][sp.where(xx[:, 0] == label)]
    agreements.append(y)
speciesName = nomenclature[1:14, 6]
box = plt.boxplot(
    agreements,
    labels=speciesName,
    patch_artist=True,
    widths=0.9,
    boxprops=boxprops)
colors = speciesColors[1:14]
for patch, color in zip(box['boxes'], colors):
    patch.set(facecolor=color, alpha=0.7)
plt.xlabel('Species')
plt.title('Number of agreement by species in 9 classification (2006-2014)')
plt.ylabel('Number of agreement')
plt.xticks(range(1, 14), speciesName, rotation=90)
plt.legend()
