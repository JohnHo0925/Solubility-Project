import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from solubilitynn_1 import GCN
import numpy as np
from solubility import solubility_pp

# torch.set_default_tensor_type('torch.LongTensor')
df = pd.read_csv('delaney_small.csv')
y_all = df['logp']
x_all = df.drop('logp',axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all)

x_train, y_train = np.array(x_train).flatten(), np.array(y_train).flatten()

def get_features(x):
    return [solubility_pp(e) for e in x]

model = GCN() # TODO

loss_fn = torch.nn.MSELoss() # is this the correct loss that you want to use?

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # feel free to play around with this

n_epochs = 10000 # or whatever
batch_size = 1 # or whatever

ctr = 0
for epoch in range(n_epochs):
    if epoch % 100 == 0:
        print(epoch)
    # X is a torch Variable
    permutation = torch.randperm(int(x_train.size))

    for i in range(0, int(x_train.size), batch_size):
        # print("i=", i)
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]

        a2b, b2a, b2revb, f_bonds, f_atoms, bond_sum = solubility_pp(x_train[indices])

        batch_y = torch.FloatTensor(np.array([y_train[indices]]))
        
        outputs = model.forward(a2b, b2a, b2revb, f_bonds, f_atoms, bond_sum)

        loss = loss_fn(outputs, batch_y)
        if ctr % 100 == 0:
            print("ctr =", ctr, "loss =", loss)
        ctr += 1
        loss.backward()
        optimizer.step()
