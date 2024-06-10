import numpy as np
import os

#IDs' of the unknown/new molecules need to train---------------
ids = np.loadtxt('correction_ids.dat', unpack=True, dtype=int)

# Ensure the array is at least 1-dimensional
ids = np.atleast_1d(ids)

#Reading CV smile string file----------------------------------
CV_smile = open('smile_string.txt', 'r')
CV_lines = CV_smile.readlines()
CV_smile.close()

#Appending unknown molecules to train smile string file--------
Train_smile = open('../Train/smile_string.txt', 'a')
#And Re-writing the CV smile file------------------------------
CV_smile = open('smile_string.txt', 'w')
for i, line in enumerate(CV_lines):
    for j in ids:
        if i == j:
            Train_smile.write('%s' %line)
    if i not in ids:
            CV_smile.write('%s' %line)
CV_smile.close()
Train_smile.close()
           
#Reading CV y values file--------------------------------------
CV_y = open('y_CV.dat', 'r')
CVy_lines = CV_y.readlines()
CV_y.close()

#Appending unknown molecules to train y vl file ----------------
Train_y = open('../Train/y_train.dat', 'a')
#And Re-writing the CV y values file ------
CV_y = open('y_CV.dat', 'w')
for i, line in enumerate(CVy_lines):
    for j in ids:
        if i == j:
            Train_y.write('%s' %line)
    if i not in ids:
        CV_y.write('%s' %line)
CV_y.close()
Train_y.close()

#Removing the correction ids to avoid repetitive correction-----
os.remove('correction_ids.dat')
