import numpy as np


#Reading feature and index file
with open('atom_index.dat', 'r') as index_file:
    Index = [[int(num) for num in line.strip().split('    ')] for line in index_file]
mol_num = len(Index)

features = np.loadtxt('features.dat')
x0 = np.ones((mol_num,1))
features = np.hstack((x0,features))

#Reading variable data---------------
y = np.loadtxt('y_train.dat', unpack=True)
parameters = np.loadtxt('parameters_new.dat', unpack=True)


#Checking we have all the parameters from the train model
if len(features[0]) == len(parameters):
    train=0

#When we have train the model
if train == 0:
    pred_y = np.zeros(mol_num)
    pred_y = np.dot(features, parameters)

outfile = open('predicted_yval.dat', 'w')
for i in range(len(pred_y)):
    outfile.write('%f\n' %pred_y[i])
outfile.close()

