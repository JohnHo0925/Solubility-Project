
import torch.nn as nn
from chemprop.nn_utils import get_activation_function, initialize_weights


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.layer = nn.Linear(316, 158)
        self.activation = nn.reLU()
        self.dropout = nn.Dropout()
        self.layer1 = nn.linear(158,1)
    def forward(self, nf_bonds):
    for c,vector in enumerate(x):
        for d in a2b[b2a[c]]:
            bond_sum = torch.add(f_bonds[d-1],bond_sum)
            bond_sum = torch.add(bond_sum,f_bonds[b2revb[c]] * -1)
            nf_bonds = torch.cat((vector,bond_sum),dim=0)
            nf_bonds = self.layer(nf_bonds)
            f_bonds[c] = nf_bonds
    molecule_pred = torch.sum(f_bonds,dim=0)
    molecule_pred = self.activation(molecule_pred)
    molecule_pred = self.dropout(molecule_pred)
    molecule_pred = self.layer1(molecule_pred)

    




