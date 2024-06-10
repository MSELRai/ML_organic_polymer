# import RDKit ----------------------------------------------------------------
import rdkit
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

# import numpy for data type conversion ---------------------------------------
import numpy as np
import pandas as pd

#Declaring variablesi----------------------------------------------------------
radi = 6    #No. of neighboring atoms (change this number as needed)

#For the smile strings need to be trained--------------------------------------
train_correction_id = []

#Converting smile string to atom id, index and bits----------------------------
def atom_indx(atom_id_file,smile_string,smile_num):
    atom_ident_r = open(atom_id_file, 'r')
    atom_id_lines = atom_ident_r.readlines()
    atom_id_line_num = len(atom_id_lines)
    atom_ident_r.close()
    
    
    # define the smiles string and covert it into a molecule sturcture----------
    smiles = smile_string
    
    mol = Chem.MolFromSmiles(smiles)
    
    #Converting smile string to Morgan Fingerprint -----------------------------
    info = {}
    fp = AllChem.GetMorganFingerprint(mol,radi,useChirality=True,bitInfo=info)
    bit_matrix = []
    
    #Reading bit info as individual components ---------------------------------
    for bit_id, atom_id_radius_tuple in info.items():
        for atom_id, radius in atom_id_radius_tuple:
            bit_matrix.append([bit_id, atom_id, radius])
    
    #Re-arranging the bit id according to radius ascending order ---------------
    bit_matrix = np.array(bit_matrix)
    bit_matrix = bit_matrix[bit_matrix[:, 2].argsort()]
    
    #Accumulating all the bit ids of an atom for different radius and making a single atom_bit id -------
    atom_num = len(mol.GetAtoms())
    atom_bit = np.zeros(atom_num)
    for i in range(atom_num):
        atom_bits = []
        for j in range(len(bit_matrix)):
            if i == bit_matrix[j][1]:
                atom_bits.append(bit_matrix[j][0])
        bit_string = [str(element) for element in atom_bits]
        atom_bit[i] = int(''.join(bit_string))
    
    #Deifining atoms new id based on accumulated bit ids -----------------------
    atom_index = []
    if len(atom_id_lines) > 0:
        for i in range(len(atom_bit)):
            found = 0
            for j in range(atom_id_line_num):
                if atom_bit[i] == int(atom_id_lines[j]):
                    atom_index.append(j)
                    found = 1
                    break
            if found == 0:
                print('We have to train model for smile no.:%d'%(smile_num))
                print(smiles)
                train_correction_id.append(smile_num)
                #writing atom bit id in ouput file
                atom_ident = open(atom_id_file, 'a')
                atom_ident.write('%d\n'%(atom_bit[i]))
                atom_ident.close()
                #appending atom index to the array
                atom_ident_r = open(atom_id_file, 'r')
                atom_id_lines = atom_ident_r.readlines()
                atom_id_line_num = len(atom_id_lines)
                atom_ident_r.close()
                atom_index.append(atom_id_line_num-1)
                train = 1

    return atom_index



#Reading smile strings from input file-------------------------------------------
smile_file = open('smile_string.txt', 'r')
smile_string_line = smile_file.readlines()
smile_string_line_num = len(smile_string_line)
smile_file.close()
atom_indexing = open('atom_index.dat', 'a')
atom_indexing.close()
f = open('atom_identity.dat', 'a')
f.close()

#Setting train=0 for considering no training is required-------------------------
train = 0

for smile_num,line in enumerate(smile_string_line):
    smile_string = line

    #Finding atom index from data file for each molecule
    mol_atom_ids=atom_indx('atom_identity.dat', smile_string, smile_num)

    #Writing the atom index of each molecule to output file
    atom_indexing = open('atom_index.dat', 'r')
    atom_index_lines = atom_indexing.readlines()
    atom_indexing.close()

    atom_indexing = open('atom_index.dat', 'a')

    if smile_num == 0 and len(atom_index_lines) == 0 and len(mol_atom_ids)>1:
        for i in range(len(mol_atom_ids)):
            atom_indexing.write('    %d'%mol_atom_ids[i])
        atom_indexing.write('\n')
    
    if smile_num > len(atom_index_lines)-1 and len(atom_index_lines) != 0:
        for i in range(len(mol_atom_ids)):
            atom_indexing.write('    %d'%mol_atom_ids[i])
        atom_indexing.write('\n')
    atom_indexing.close()

#Generating features for each molecules-----------------------------------------------
atom_ident_r = open('atom_identity.dat', 'r')
atom_id_lines = atom_ident_r.readlines()
atom_id_line_num = len(atom_id_lines)
atom_ident_r.close()

features = np.zeros((smile_string_line_num, atom_id_line_num), dtype=float)
with open('atom_index.dat', 'r') as index_file:
    Index = [[int(num) for num in line.strip().split('    ')] for line in index_file]

for i in range(len(Index)):
    for j in Index[i]:
        features[i][j] = 1

#Writing out the feature in a file for ML--------------------------------------------
feature_file = open('features.dat', 'w')
for i in range(len(features)):
    for j in range(len(features[i])):
        feature_file.write('    %f' %features[i][j])
    feature_file.write('\n')
feature_file.close()


#Writing out the index of the smile string(s) need(s) to be trained---------
if len(train_correction_id) > 0:
    #Deleting the repeating ids from----------------------------------------
    unique_correction_id = sorted(list(set(train_correction_id)))
    train_id_file = open('correction_ids.dat', 'w')
    for i in unique_correction_id:
        train_id_file.write('%d\n' %i)
    train_id_file.close()
