from __future__ import print_function
from rdkit import Chem
import numpy as np
import torch
import pdb
 
def solubility_pp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    f_atoms = []  # mapping from atom index to atom features
    f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
    a2b = []  # mapping from atom index to incoming bond indices
    b2a = []  # mapping from bond index to the index of the atom the bond is coming from```
    b2revb = []
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

    def one_hotyk(val):
        if val is True:
            encoding = [1,0]
        else:
            encoding= [0,1]
        return encoding

    def atom_features(atom):
        encoding_num = one_hot(atom.GetAtomicNum(), atomic_num)

        encoding_degree = one_hot(atom.GetTotalDegree(), degree)

        encoding_charge = one_hot(atom.GetFormalCharge(), formal_charge)

        encoding_chiral = one_hot(atom.GetChiralTag(), chiral_tag)

        encoding_Hs = one_hot(atom.GetTotalNumHs(), num_Hs)

        encoding_hybrid = one_hot(atom.GetTotalNumHs(), hybridization)

        encoding_aromatic = one_hotyk(atom.GetIsAromatic())

        atom_feature = encoding_num + encoding_degree + encoding_charge + encoding_chiral + encoding_Hs + encoding_hybrid + encoding_aromatic
        return atom_feature
          
    def bond_features(bond):

        encoding_bondtype = one_hot(bond.GetBondType(), bond_type)

        encoding_conjugation = one_hotyk(bond.GetIsConjugated())

        encoding_stereo = one_hotyk(bond.GetStereo())

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
    b = 0


    for x in range(int(len(f_bonds)/2)):
        a = b+1
        b2revb.append(a)
        b2revb.append(b)
        b += 2

    f_atoms = torch.LongTensor(f_atoms)
    f_bonds = torch.LongTensor(f_bonds)
    b2a = torch.LongTensor(b2a)
    b2revb = torch.LongTensor(b2revb)

    max1 = 0
    bond_sum = [0]* 158
    bond_sum = torch.LongTensor(bond_sum)

    for list2 in a2b:
        if len(list2) > max1:
           max1 = len(list2)

    for i, list2 in enumerate(a2b):
        if len(list2) < max1:
           a2b[i] = list2 + [0] * (max1-len(list2))

    a2b = torch.LongTensor(a2b)
    # print(b2revb)

    return a2b, b2a, b2revb, f_bonds, f_atoms, bond_sum

