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

class FGIGattack():

    def __init__(self, model, nnodes=None, feature_shape=None, attack_structure=True, attack_features=False, device='cpu'):
        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.attack_structure = attack_structure
        self.device = device
        self.modified_adj = None
        self.modified_features = None
        self.target_node = None
        self.running_time = None

    def attack(self, model, ori_features, ori_adj, labels, idx_train, target_node, n_perturbations, steps=10, m=2,  **kwargs):
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

        index_list = self.grad_max_index(model, features, adj_norm, n_perturbations, m)
        # print(index_list[1])
        # print(index_list.shape[0])
        s_e = np.zeros(adj.shape[1])

        if self.attack_structure:
            s_e = self.calc_importance_edge(victim_model, features, adj_norm, labels, steps, index_list)
            # print(s_e)

        for t in (range(n_perturbations)):
            s_e_max = np.argmax(s_e)
            # print(s_e_max)
            max_e_index = index_list[s_e_max]
            # print(max_e_index)
            if self.attack_structure:
                value = np.abs(1 - modified_adj[target_node, max_e_index])
                modified_adj[target_node, max_e_index] = value
                s_e[s_e_max] = 0
            else:
                raise Exception("""No posisble perturbation on the structure can be made!
                        See https://github.com/DSE-MSU/DeepRobust/issues/42 for more details.""")
        self.modified_adj = modified_adj
        end = time.perf_counter()
        Running_time = end - start
        self.running_time = Running_time


    def grad_max_index(self, model, features, adj_norm, n_perturbations, m):
        adj_norm.requires_grad = True
        output = model.forward(features, adj_norm)
        loss = F.nll_loss(output[[self.target_node]], self.pseudo_labels[[self.target_node]])
        grad = torch.autograd.grad(loss, adj_norm)[0]
        target_grad = grad[self.target_node]
        for e in range(target_grad.shape[0]):
            target_grad[e] = torch.abs(target_grad[e])
        # print(target_grad)
        if n_perturbations*m > 10:
            seed_index_num = n_perturbations*m
        else:
            seed_index_num = 10
        index_list = target_grad.argsort()[-seed_index_num:]
        # print(np.argmax(target_grad))
        # print(index_list)
        return index_list




    def calc_importance_edge(self, model, features, adj_norm, labels, steps, index_list):
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
        for j in range(index_list.shape[0]):
            edge_index = index_list[j]
            # print(edge_index)
            if adj_norm[i][edge_index]:
                scaled_inputs = [baseline_remove + (float(k)/ steps) * (adj_norm - baseline_remove) for k in range(0, steps + 1)]
            else:
                scaled_inputs = [baseline_add - (float(k)/ steps) * (baseline_add - adj_norm) for k in range(0, steps + 1)]
            _sum = 0

            for new_adj in scaled_inputs:
                output = model.forward(features, new_adj)
                loss = F.nll_loss(output[[self.target_node]],
                        self.pseudo_labels[[self.target_node]])
                adj_grad = torch.autograd.grad(loss, adj_norm)[0]
                adj_grad = adj_grad[i][edge_index]
                _sum += adj_grad

            if adj_norm[i][edge_index]:
                # avg_grad = (adj_norm[i][edge_index] - 0) * _sum.mean()
                avg_grad = _sum.mean()
            else:
                avg_grad = _sum.mean()
                # avg_grad = (1 - adj_norm[i][edge_index]) * _sum.mean()
            # avg_grad = torch.abs(avg_grad)
            integrated_grad_list.append(avg_grad.detach().item())

        integrated_grad_list = np.array(integrated_grad_list)
        return integrated_grad_list
