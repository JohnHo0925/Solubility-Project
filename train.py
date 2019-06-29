import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from solubilitynn import GCN
import numpy as np
from solubility import solubility_pp

# torch.set_default_tensor_type('torch.LongTensor')
df = pd.read_csv('delaney.csv')
y_all = df['logp']
x_all = df.drop('logp',axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all)

x_train, y_train = np.array(x_train).flatten(), np.array(y_train).flatten()

def get_features(x):
    return [solubility_pp(e) for e in x]

model = GCN() 

loss_fn = torch.nn.MSELoss() \

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100 # or whatever
batch_size = 13 # or whatever

ctr = 0
losses = []
for epoch in range(n_epochs):
    print(epoch)
    permutation = torch.randperm(int(x_train.size))
    for i in range(0, int(x_train.size), batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        a2b = []
        b2a = []
        b2revb = []
        f_bonds = []
        f_atoms = []
        bond_sum = []
        for x in get_features(x_train[indices]):
             a2b.append(x[0]),b2a.append(x[1]),b2revb.append(x[2]),f_bonds.append(x[3]),f_atoms.append(x[4]),bond_sum.append(x[5])
        batch_y = torch.FloatTensor(np.array([y_train[indices]]))
        
        outputs = model.forward(a2b, b2a, b2revb, f_bonds, f_atoms, bond_sum)
            # print(x_train[indices])
        loss = loss_fn(outputs, batch_y[0])
        losses.append(loss)

        if ctr % 100 == 0:
            print("ctr =", ctr, "loss =", loss)
        ctr += 1

        loss.backward()
        optimizer.step()

def predict(smile):
    a2b, b2a, b2revb, f_bonds, f_atoms, bond_sum = solubility_pp(smile)
    return model.forward(a2b, b2a, b2revb, f_bonds, f_atoms, bond_sum)