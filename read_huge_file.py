import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

df = pd.read_csv('huge_file.csv')
solubility = df['solubility ALOGPS']
smile_string = df['smiles']
z = []
correct_smiles = []
for x, y in zip(solubility, smile_string):
	mol = Chem.MolFromSmiles(y)
	if mol is not None:
		z.append(np.log(float(x)/Descriptors.MolWt(mol)))
		correct_smiles.append(y)

with open('more_data.csv', 'w') as f:
	for a, b in zip(smile_string, z):
		print(a, b, file=f, sep=",")

