import numpy as np
import os

#IDs' of the unknown/new molecules need to train---------------
ids = np.loadtxt('correction_ids.dat', unpack=True, dtype=int)

# Ensure the array is at least 1-dimensional
ids = np.atleast_1d(ids)

#Reading CV smile string file----------------------------------
test_smile = open('smile_string.txt', 'r')
test_lines = test_smile.readlines()
test_smile.close()

#Appending unknown molecules to train smile string file--------
Train_smile = open('../Train/smile_string.txt', 'a')
#And Re-writing the test smile file------------------------------
test_smile = open('smile_string.txt', 'w')
for i, line in enumerate(test_lines):
    for j in ids:
        if i == j:
            Train_smile.write('%s' %line)
    if i not in ids:
            test_smile.write('%s' %line)
test_smile.close()
Train_smile.close()
           
#Reading test y values file--------------------------------------
test_y = open('y_test.dat', 'r')
testy_lines = test_y.readlines()
test_y.close()

#Appending unknown molecules to train y vl file ----------------
Train_y = open('../Train/y_train.dat', 'a')
#And Re-writing the CV y values file ------
test_y = open('y_test.dat', 'w')
for i, line in enumerate(testy_lines):
    for j in ids:
        if i == j:
            Train_y.write('%s' %line)
    if i not in ids:
        test_y.write('%s' %line)
test_y.close()
Train_y.close()

#Removing the correction ids to avoid repetitive correction-----
os.remove('correction_ids.dat')
