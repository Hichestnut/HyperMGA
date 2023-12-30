"""
    Adversarial Examples on Graph Data: Deep Insights into Attack and Defense
        https://arxiv.org/pdf/1903.01610.pdf
"""

import torch
import torch.multiprocessing as mp

from torch.nn.parameter import Parameter
from deeprobust.graph import utils
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from torch import optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
import numpy as np
from tqdm import tqdm
import math
import scipy.sparse as sp
import utils.hypergraph_utils as hgut
import time

class IGattack():

    def __init__(self, model, nnodes=None, feature_shape=None, attack_structure=True, attack_features=True, device='cpu'):
        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.attack_structure = attack_structure
        self.device = device
        self.modified_adj = None
        self.modified_features = None
        self.target_node = None
        self.running_time = None

    def attack(self, model, ori_features, ori_adj, labels, idx_train, target_node, n_perturbations, steps=10, **kwargs):
        start = time.perf_counter()
        victim_model = model
        self.target_node = target_node

        modified_adj = ori_adj.copy()
        ori_features = torch.Tensor(ori_features).to('cpu')
        labels = torch.Tensor(labels).to('cpu')
        adj, features, labels = utils.to_tensor(modified_adj, ori_features, labels, device='cpu')
        adj_norm = hgut.generate_G_from_H(adj)
        adj_norm = torch.Tensor(adj_norm).to(self.device)
        features = torch.Tensor(features).to(self.device)
        labels = torch.Tensor(labels).to(self.device)
        ori_result = victim_model.forward(features, adj_norm)
        pseudo_labels = ori_result.detach().argmax(1)
        pseudo_labels[idx_train] = labels[idx_train]
        self.pseudo_labels = pseudo_labels

        s_e = np.zeros(adj.shape[1])

        if self.attack_structure:
            s_e = self.calc_importance_edge(victim_model, features, adj_norm, labels, steps)

        for t in (range(n_perturbations)):
            s_e_max = np.argmax(s_e)

            if self.attack_structure:
                value = np.abs(1 - modified_adj[target_node, s_e_max])
                modified_adj[target_node, s_e_max] = value
                s_e[s_e_max] = 0
            else:
                raise Exception("""No posisble perturbation on the structure can be made!
                        See https://github.com/DSE-MSU/DeepRobust/issues/42 for more details.""")
        self.modified_adj = modified_adj
        end = time.perf_counter()
        Running_time = end-start
        self.running_time = Running_time

    def calc_importance_edge(self, model, features, adj_norm, labels, steps):
        """Calculate integrated gradient for edges. Although I think the the gradient should be
        with respect to adj instead of adj_norm, but the calculation is too time-consuming. So I
        finally decided to calculate the gradient of loss with respect to adj_norm
        """
        baseline_add = adj_norm.clone()
        baseline_remove = adj_norm.clone()
        baseline_add.data[self.target_node] = 1
        baseline_remove.data[self.target_node] = 0
        adj_norm.requires_grad = True
        integrated_grad_list = []

        i = self.target_node
        for j in range(adj_norm.shape[1]):
            if adj_norm[i][j]:
                scaled_inputs = [baseline_remove + (float(k)/ steps) * (adj_norm - baseline_remove) for k in range(0, steps + 1)]
            else:
                scaled_inputs = [baseline_add - (float(k)/ steps) * (baseline_add - adj_norm) for k in range(0, steps + 1)]
            _sum = 0

            for new_adj in scaled_inputs:
                output = model.forward(features, new_adj)
                loss = F.nll_loss(output[[self.target_node]],
                        self.pseudo_labels[[self.target_node]])
                adj_grad = torch.autograd.grad(loss, adj_norm)[0]
                adj_grad = adj_grad[i][j]
                _sum += adj_grad

            if adj_norm[i][j]:
                avg_grad = (adj_norm[i][j] - 0) * _sum.mean()
            else:
                avg_grad = (1 - adj_norm[i][j]) * _sum.mean()

            integrated_grad_list.append(avg_grad.detach().item())

        integrated_grad_list[i] = 0
        # make impossible perturbation to be negative
        integrated_grad_list = np.array(integrated_grad_list)
        adj = (adj_norm > 0).cpu().numpy()
        integrated_grad_list = (-2 * adj[self.target_node] + 1) * integrated_grad_list
        integrated_grad_list[self.target_node] = -10
        return integrated_grad_list



