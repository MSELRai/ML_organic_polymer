import numpy as np


#Reading feature and index file------------------------------------
with open('atom_index.dat', 'r') as index_file:
    Index = [[int(num) for num in line.strip().split('    ')] for line in index_file]
mol_num = len(Index)

#Adding constant to features for (a = bx + c; c=constant)----------
features = np.loadtxt('features.dat')
x0 = np.ones((mol_num,1))
features = np.hstack((x0,features))

#Reading variable data---------------------------------------------
y = np.loadtxt('y_test.dat', unpack=True)
parameters = np.loadtxt('parameters.dat', unpack=True)


#Checking we have all the parameters from the train model----------
if len(features[0]) == len(parameters):
    train=0

#When we have trained model----------------------------------------
if train == 0:
    pred_y = np.zeros(mol_num)
    pred_y = np.dot(features, parameters)

#Writing out the predicted properties------------------------------
outfile = open('predicted_yval.dat', 'w')
for i in range(len(pred_y)):
    outfile.write('%f\n' %pred_y[i])
outfile.close()

