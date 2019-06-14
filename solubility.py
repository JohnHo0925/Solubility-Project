from __future__ import print_function
from rdkit import Chem
import numpy as np
import torch
smiles = 'CC(NC1=NN=C(S1)[S](N)(=O)=O)=O'
mol = Chem.MolFromSmiles(smiles)
f_atoms = []  # mapping from atom index to atom features
f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
a2b = []  # mapping from atom index to incoming bond indices
b2a = []  # mapping from bond index to the index of the atom the bond is coming from```
max_num = 115
atomic_num = list(range(max_num))
degree = [0, 1, 2, 3, 4, 5]
formal_charge = [-1, -2, 1, 2, 0]
chiral_tag = [0, 1, 2, 3]
num_Hs = [0, 1, 2, 3, 4]
hybridization = [
        0,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ]
bond_type = [
            Chem.rdchem.BondType.SINGLE,
          	Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]

def one_hot(val, list1):
    encoding = [0] * (len(list1) + 1)
    if val in list1:
        index = list1.index(val)
    else:
        index = -1
    encoding[index] = 1
    return encoding
 
#atomfeature = []
#encoding_num = one_hot(atom.GetAtomicNum(),atomic_num)
#encoding_num = [0] * (len(atomic_num) + 1)    


def atom_features(atom):
    encoding_num = [0] * (len(atomic_num) + 1)
    if atom.GetAtomicNum() in atomic_num:
    	index = atomic_num.index(atom.GetAtomicNum())
    else:
    	index = -1
    encoding_num[index] = 1
   
    encoding_degree = [0] * (len(degree) + 1) 
    if atom.GetTotalDegree() in degree:
    	index = degree.index(atom.GetTotalDegree())
    else:
    	index = -1
    encoding_degree[index] = 1   
    
    encoding_charge = [0] * (len(formal_charge) + 1)
    if atom.GetFormalCharge() in formal_charge:
    	index = formal_charge.index(atom.GetFormalCharge())
    else:
    	index = -1
    encoding_charge[index] = 1 
    
    encoding_chiral = [0] * (len(chiral_tag) + 1)
    if atom.GetChiralTag() in chiral_tag:
    	index = chiral_tag.index(atom.GetChiralTag())
    else:
    	index = -1
    encoding_chiral[index] = 1 
    
    encoding_Hs = [0] * (len(num_Hs) + 1)
    if atom.GetTotalNumHs() in num_Hs:
    	index = num_Hs.index(atom.GetTotalNumHs())
    else:
    	index = -1
    encoding_Hs[index] = 1 
    
    encoding_hybrid = [0] * (len(hybridization) + 1)
    if atom.GetHybridization() in hybridization:
    	index = hybridization.index(atom.GetHybridization())
    else:
    	index = -1
    
    encoding_hybrid[index] = 1
    if atom.GetIsAromatic() is True:
    	encoding_aromatic = [1,0]
    else:
    	encoding_aromatic = [0,1]
    
    atom_feature = encoding_num + encoding_degree + encoding_charge + encoding_chiral + encoding_Hs + encoding_hybrid + encoding_aromatic
    return atom_feature

def bond_features(bond):
    
    encoding_bondtype = [0] * (len(bond_type) + 1)
    
    if bond.GetBondType() in bond_type:
    	index = bond_type.index(bond.GetBondType())
    else:
    	index = -1
    encoding_bondtype[index] =  1
   
    if bond.GetIsConjugated() is True:
    	encoding_conjugation = [1,0]
    else:
    	encoding_conjugation = [0,1]
   
    if bond.GetStereo() is True:
    	encoding_stereo = [1,0]
    else:
        encoding_stereo = [0,1]
    
    bond_feature = encoding_bondtype + encoding_conjugation + encoding_stereo
    return bond_feature

for x in range(mol.GetNumAtoms()):
    a2b.append([])

for g, atom in enumerate(mol.GetAtoms()):
    f_atoms.append(atom_features(atom))
f_atoms = [f_atoms[g] for g in range(mol.GetNumAtoms())]

bond_index = 1
for x1 in range(mol.GetNumAtoms()):
    for x2 in range(x1 + 1, mol.GetNumAtoms()):
        bond = mol.GetBondBetweenAtoms(x1, x2)

        if bond is None:
            continue

        f_bond = bond_features(bond)
                
        f_bonds.append(f_atoms[x1] + f_bond)
        f_bonds.append(f_atoms[x2] + f_bond)

        y1 = bond_index 
       	y2 = y1 + 1

        a2b[x1].append(y1)
        a2b[x2].append(y2)

        b2a.append(x1)
       	b2a.append(x2)
        bond_index += 2



f_atoms = torch.LongTensor(f_atoms)
f_bonds = torch.LongTensor(f_bonds)
b2a = torch.LongTensor(b2a)

max1 = 0
for list2 in a2b:
	if len(list2) > max1:
		max1 = len(list2)
for i, list2 in enumerate(a2b):
	if len(list2) < max1:
		 a2b[i] = list2 + [0] * (max1-len(list2))

a2b = torch.LongTensor(a2b)

print(f_atoms)
print(f_bonds)
print(b2a)
print(a2b)



 