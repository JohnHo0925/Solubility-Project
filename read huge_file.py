import pandas as pd
from rdkit import Chem
import numpy as np
df = pd.read_csv('huge_file.csv')
solubility = df['solubility ALOGPS']
smile_string = df['smiles']
z = []

for x, y in zip(solubility, smile_string):
	z.append(np.log(float(x)/MolWt(Chem.MolFromSmiles(y))))

with open('more_data.csv', 'w') as f:
	for a, b in zip(smile_string, z):
		print(a, b, file=f)

