import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
matplotlib.rcParams['axes.linewidth'] = 1.5


#Reading feature and index file--------------------------------------------------
with open('atom_index.dat', 'r') as index_file:
    Index = [[int(num) for num in line.strip().split('    ')] for line in index_file]
mol_num = len(Index)

#Reading features and adding 1 as constant (a = bx + c; c=constanct)--------------
features = np.loadtxt('features.dat')
x0 = np.ones((mol_num,1))
features = np.hstack((x0,features))

#Reading output variable data-----------------------------------------------------
y = np.loadtxt('y_train.dat', unpack=True)
y1 = np.loadtxt('../CV/y_CV.dat', unpack=True)

#Repeating For CV-----------------------------------------------------------------
features1 = np.loadtxt('../CV/features.dat')
x1 = np.ones((len(y1),1))
features1 = np.hstack((x1,features1))


#Declaring variables and change the values as needed
iteration = 1000    #No. of iteration for ML
alpha = 0.1         #learning rate
lamda = 10          #regularization term


#Regularized linear regression model-----------------------
def RegLinReg(y, pred_y, features, parameters, alpha, lamda, mol_num):
    residual = np.subtract(pred_y, y)

    #Creating temporary parameters with 1st element=0 for intercept regularization-------
    temp_par = parameters
    temp_par[0] = 0

    parameters = parameters - (alpha/mol_num) * ((features.T @ residual) + (lamda * temp_par))
    return parameters



#Cost function----------------------------------------------------------------------------
def cost_function(y, pred_y, mol_num, lamda, parameters, y1, pred_y1):
    residual = np.subtract(pred_y, y)
    residual1 = np.subtract(pred_y1, y1)
    cost = (1/(2*mol_num)) * (np.dot(residual,residual) +  (lamda * np.dot(parameters, parameters)))
    cost1 = (1/(2*mol_num)) * (np.dot(residual1,residual1) +  (lamda * np.dot(parameters, parameters)))

    return cost,cost1



#Generating parameters for new atoms and unknown features----------------------
feature_num = len(features[0])

parameter_file = open('parameters.dat', 'a')
parameter_file.close()
parameter_file = open('parameters.dat', 'r')
parameter_line = parameter_file.readlines()
parameter_file.close()



#Generating intial random parameters with gaussian distribution----------------
if len(parameter_line) == 0:
    print('We have to train the model')
    train = 1

    np.random.seed(1234567890)
    parameters= np.random.normal(0, 0.30, feature_num)

#Reading previoulsy created parameters-----------------------------------------
if len(parameter_line) != 0:
    print('We have to train the model')
    train = 1

    parameters = np.loadtxt('parameters.dat', unpack=True)


#Starting to train the model----------------------------------------------------
if train == 1:
    pred_y = np.zeros(mol_num)
    pred_y = np.dot(features, parameters)

#Training the regularized regression model--------------------------------------
J = np.zeros(iteration)
J_cv = np.zeros(iteration)
x = np.zeros(iteration)
print('Model training is ongoing ......')
for niter in range(iteration):
    parameters = RegLinReg(y, pred_y, features, parameters, alpha, lamda, mol_num)
    pred_y = np.dot(features, parameters)
    #For CV --------------------------------------------------------------------
    pred_y1 = np.dot(features1, parameters)
    
    #Calculating cost-----------------------------------------------------------
    J[niter], J_cv[niter] = cost_function(y, pred_y, mol_num, lamda, parameters, y1, pred_y1)
    x[niter] = niter + 1

#Writing the trained parameters in output file----------------------------------
parameter_file = open('parameters_new.dat', 'w')
for i in range(len(parameters)):
    parameter_file.write('    %f\n' %parameters[i])
parameter_file.close()

#Plot cost function-------------------------------------------------------------
fig, ax = plt.subplots()
ax.plot(x, J, lw=2, color='darkred', label = r'$Train$')
ax.plot(x, J_cv, lw=2, color='darkblue', label = r'$CV$')
ax.tick_params(axis='both', which='major', labelsize=17)
ax.tick_params(axis='both', which='minor', labelsize=17)
ax.set_ylabel(r'$J(\theta)$', fontsize=20)
ax.set_xlabel(r'$\mathrm{No.~of~Iterations}$', fontsize=20)
ax.yaxis.set_minor_locator(AutoMinorLocator(1))
ax.xaxis.set_minor_locator(AutoMinorLocator(1))
plt.legend()
plt.savefig('iteration_vs_cost.png', dpi=720, bbox_inches = "tight")
plt.show()
