import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from solubilitynn import GCN
import numpy as np
from solubility import solubility_pp
import pdb

# torch.set_default_tensor_type('torch.LongTensor')
df = pd.read_csv('current_data.csv')
y_all = df['logp']
x_all = df.drop('logp',axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all,random_state = 42)

x_train, y_train = np.array(x_train).flatten(), np.array(y_train).flatten()
x_test, y_test = list(np.array(x_test).flatten()), list(np.array(y_test).flatten()) # lists
y_test = torch.FloatTensor(np.array([y_test]))[0]
def get_features(x):
    return [solubility_pp(e) for e in x]

model = GCN() 

loss_fn = torch.nn.MSELoss()


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 100 # or whatever
batch_size = 10 # or whatever

# smile_batch: list of strings where each string is a smile string describing one of the molecules in the batch. 
#     list is length N, where N is the batch size.
def predict(smile_batch):
    batch_size = len(smile_batch)
    a2b = []
    b2a = []
    b2revb = []
    f_bonds = []
    f_atoms = []
    bond_sum = []
    for x in get_features(smile_batch):
        a2b.append(x[0]),b2a.append(x[1]),b2revb.append(x[2]),f_bonds.append(x[3]),f_atoms.append(x[4]),bond_sum.append(x[5])
    outputs = model.forward(a2b, b2a, b2revb, f_bonds, f_atoms, bond_sum)
    return outputs


ctr = 0
losses = []
for epoch in range(n_epochs):
    print(epoch)
    permutation = torch.randperm(int(x_train.size))
    for i in range(0, int(x_train.size), batch_size):
        optimizer.zero_grad()
        try:
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
            loss = torch.sqrt(loss_fn(outputs, batch_y[0]))
            if loss.item() > 1000:
                pdb.set_trace()
            losses.append(loss)

            if ctr % 100 == 0:
                print("ctr =", ctr, "loss =", loss)
            ctr += 1

            loss.backward()
            optimizer.step()
            if epoch%10 == 0:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, "checkpoint" + str(epoch))
        except:
            print(x_train[indices])

