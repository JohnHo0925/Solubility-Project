
import torch.nn as nn
import torch


class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.layer = nn.Linear(316, 158)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer1 = nn.Linear(158,110)
        self.layer2 = nn.Linear(110,70)
        self.layer3 = nn.Linear(70,35)
        self.layer4 = nn.Linear(35,1)

    def forward(self, a2b_, b2a_, b2revb_, f_bonds_, f_atoms_, bond_sum_):
        molecule_pred_ = torch.Tensor()
        for a2b, b2a, b2revb, f_bonds, f_atoms, bond_sum in zip(a2b_, b2a_, b2revb_, f_bonds_, f_atoms_, bond_sum_):
            for c,vector in enumerate(f_bonds):
                for d in a2b[b2a[c]]:
                    bond_sum = f_bonds[d-1] + bond_sum
                    bond_sum = bond_sum - f_bonds[int(b2revb[c])]
                    nf_bonds = (torch.cat((vector,bond_sum),dim=0)).float()
                    nf_bonds = self.layer(nf_bonds)
                    f_bonds[c] = nf_bonds
            molecule_pred = torch.sum(f_bonds,dim=0)
            molecule_pred = self.dropout(molecule_pred.float())
            molecule_pred = self.layer1(molecule_pred)
            molecule_pred = self.dropout(molecule_pred.float())
            molecule_pred = self.layer2(molecule_pred)
            molecule_pred = self.activation(molecule_pred)
            molecule_pred = self.dropout(molecule_pred.float())
            molecule_pred = self.layer3(molecule_pred)
            molecule_pred = self.activation(molecule_pred)
            molecule_pred = self.dropout(molecule_pred.float())
            molecule_pred = self.layer4(molecule_pred)
            molecule_pred_ = torch.cat((molecule_pred_, molecule_pred))
        return molecule_pred_
