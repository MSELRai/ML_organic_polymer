#Here we demonstrate the prediction of electronic gap using r=6 (r = No. of neighboring atoms to identify an atom)

#The prediction pocess three main steps- training, cross validation (CV) and testing.

#These three steps are done in three folders named Train, CV, and testData

#The data needs to randomly splitted among these folders maintaining the ratio you wanted

#For each step smile strings need to be supplied in a file with a filename smile_string.txt

#The property which is going to be predicted provided with filename y_train.dat, y_CV.dat, and y_test.dat, respectively

#The values of property should be in the same order as the smile strings are.

#First direct to Train folder
cd Train

#Then run MorganFingerP.py to generate atom library for the Train dataset (smile strings)
python MorganFingerP.py

#Run MorganFingerP.py again to generate the features for each smile string
python MorganFingerP.py

#Then copy the library ('atom_identity.dat' file) to CV folder to create features for CV dataset
cp atom_identity.dat ../CV/.
cd ../CV/
python MorganFingerP.py

#If the code finds new atom type in the dataset that is not available in the library we need add that atom type to training dataset. For that we need to add the smile string to train dataset which has new atom type. The code will write out the index of smiles of strings need to be trained in 'correction_ids.dat' file.

#We need to run 'train_correction.py' only if it finds new atom types
python train_correction.py

#The remove the previously generated atom library from Test and CV folders
rm atom_identity.dat features.dat
cd ../Train/
rm atom_identity.dat features.dat

#Then generate the new atom library by running 'MorganFingerP.py' twice as before
python MorganFingerP.py
python MorganFingerP.py

#We also need to check for new atom type in the train dataset
cp atom_identity.dat ../testData/.
cd ../testData/
python MorganFingerP.py

#We need to run 'train_correction.py' only if it finds new atom types
python train_correction.py

#The remove the previously generated atom library from Test and CV folders
rm atom_identity.dat features.dat
cd ../Train/
rm atom_identity.dat features.dat

#Then generate the new atom library by running 'MorganFingerP.py' twice as before
python MorganFingerP.py
python MorganFingerP.py

#Copy the library to CV and testData
cp atom_identity.dat ../CV/.
cp atom_identity.dat ../testData/.

#Generate features for CV for test dataset
cd ../CV/
python MorganFingerP.py
cd ../testData/
python MorganFingerP.py

#Now train the model (adjust the no. of iteration and hyper parameters)
cd ../Train/
python ML_train.py

#This will generate the parameters for trained model in 'parameters_new.dat' file. If the cost of the training seems good, copy parameters_new.dat to CV and test folder as parameters.dat
cp parameters_new.dat ../CV/parameters.dat
cp parameters_new.dat ../testData/parameters.dat

#Then predict the properties for CV and test dataset
cd ../CV/
python ML_CV.py
cd ../testData/
python ML_test.py

#This will calcualte the properties and write out the predicted propeties in predicted_yval.dat

#These predicted properties are used for statistical analysis.
