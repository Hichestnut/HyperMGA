from torch import nn
from models.layers import HGNN_conv
import torch.nn.functional as F
import utils.hypergraph_utils as hgut
import torch
import numpy as np

class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x

    def forward_H(self, x, H):
        # H = H.detach().numpy()
        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = torch.ones(n_edge)
        # the degree of the node
        DV = torch.sum(H * W)
        # the degree of the hyperedge
        DE = torch.sum(H)

        invDE = torch.tensor(torch.diag(torch.pow(torch.tensor([DE]), torch.tensor([-1]))), device='cpu')
        DV2 = torch.tensor(torch.diag(torch.pow(torch.tensor([DV]), torch.tensor([-0.5]))), device='cpu')
        # DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        W = torch.tensor(torch.diag(W), device='cpu')
        H = torch.tensor(H, device='cpu')
        # H = np.mat(H)
        HT = H.t()
        G = DV2 * H * W * invDE * HT * DV2
        # G = hgut._generate_G_from_H(H, variable_weight=False)
        # G = torch.Tensor(G).to('cpu')
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x
