import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
import utils.hypergraph_utils as hgut
from deeprobust.graph import utils
from deeprobust.graph.global_attack import PGDAttack

def PGDattack(model,ori_features, ori_adj, labels, idx_train, n_perturbations, epochs=200, loss_type='CE'):
    adj_changes = Parameter(torch.FloatTensor(int(ori_features.shape[0] * (ori_features.shape[0] - 1) / 2)))
    adj_changes.data.fill_(0)
    victim_model = model
    ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels)

    print(type(adj_changes))
    for t in tqdm(range(epochs)):
        modified_adj = get_modified_adj(ori_adj, ori_features.shape[0], adj_changes)
        print(modified_adj)
        modified_adj = modified_adj.detach().numpy()
        modified_G = hgut._generate_G_from_H(modified_adj, variable_weight=False)
        # modified_adj.device = 'cpu'
        modified_G = torch.Tensor(modified_G).to('cpu')

        adj_norm = utils.normalize_adj_tensor(modified_G)
        output = victim_model.forward(ori_features, adj_norm)
        adj_norm.requires_grad = True
        # loss = F.nll_loss(output[idx_train], labels[idx_train])
        loss = _loss(output[idx_train], labels[idx_train], loss_type)
        print(loss)
        # print(adj_changes)
        adj_grad = torch.autograd.grad(loss, adj_norm)[0]
        if loss_type == 'CE':
            lr = 200 / np.sqrt(t + 1)
            print(lr)
            print(adj_grad)
            adj_changes.data.add_(lr * adj_grad)

        if loss_type == 'CW':
            lr = 0.1 / np.sqrt(t + 1)
            adj_changes.data.add_(lr * adj_grad)

        adj_changes = projection(adj_changes, n_perturbations)

    adj_changes = random_sample(victim_model, ori_adj, ori_features, labels, idx_train, n_perturbations)
    modified_adj = get_modified_adj(ori_adj, ori_features.shape[0], adj_changes).detach()
    modified_G = hgut._generate_G_from_H(modified_adj, variable_weight=False)
    G_norm = utils.normalize_adj_tensor(modified_G)
    # self.check_adj_tensor(self.modified_adj)
    return G_norm


def get_modified_adj(ori_adj, nnodes, adj_changes, complementary=None, device='cpu'):

    if complementary is None:
        complementary = (torch.ones_like(ori_adj) - torch.eye(nnodes).to(device) - ori_adj) - ori_adj
        # complementary 大小与邻接矩阵相同，已连接的节点间为-1，未连接的节点间为1

    m = torch.zeros((nnodes, nnodes)).to(device) # 大小为N × N ，值为0
    # row(int) -二维矩阵中的行数  col(int) -二维矩阵中的列数 offset(int) -与主对角线的对角线偏移。
    tril_indices = torch.tril_indices(row=nnodes, col=nnodes, offset=-1)  # 对角线偏移后的索引
    m[tril_indices[0], tril_indices[1]] = adj_changes
    m = m + m.t()
    modified_adj = complementary * m + ori_adj

    return modified_adj

def random_sample(model, adj_changes, ori_adj, ori_features, labels, idx_train, n_perturbations):
    K = 20
    best_loss = -1000
    victim_model = model

    with torch.no_grad():
        s = adj_changes.cpu().detach().numpy()
        for i in range(K):
            sampled = np.random.binomial(1, s)

            # print(sampled.sum())
            if sampled.sum() > n_perturbations:
                continue
            adj_changes.data.copy_(torch.tensor(sampled))
            modified_adj = get_modified_adj(ori_adj, ori_features.shape[0], adj_changes)
            modified_G = hgut._generate_G_from_H(modified_adj, variable_weight=False)
            adj_norm = utils.normalize_adj_tensor(modified_G)
            # adj_norm = utils.normalize_adj_tensor(modified_adj)
            output = victim_model(ori_features, adj_norm)
            loss = _loss(output[idx_train], labels[idx_train])
            # loss = F.nll_loss(output[idx_train], labels[idx_train])
            # print(loss)
            if best_loss < loss:
                best_loss = loss
                best_s = sampled
        adj_changes.data.copy_(torch.tensor(best_s))
    return adj_changes

def _loss(output, labels, loss_type="CE"):
    if loss_type == "CE":
        loss = F.nll_loss(output, labels)
    if loss_type == "CW":
        onehot = utils.tensor2onehot(labels)
        best_second_class = (output - 1000*onehot).argmax(1)
        margin = output[np.arange(len(output)), labels] - \
                output[np.arange(len(output)), best_second_class]
        k = 0
        loss = -torch.clamp(margin, min=k).mean()
        # loss = torch.clamp(margin.sum()+50, min=k)
    return loss

def projection(adj_changes, n_perturbations):
    # projected = torch.clamp(self.adj_changes, 0, 1)
    if torch.clamp(adj_changes, 0, 1).sum() > n_perturbations:
        left = (adj_changes - 1).min()
        right = adj_changes.max()
        miu = bisection(adj_changes, left, right, n_perturbations, epsilon=1e-5)
        adj_changes.data.copy_(torch.clamp(adj_changes.data - miu, min=0, max=1))
    else:
        adj_changes.data.copy_(torch.clamp(adj_changes.data, min=0, max=1))
    return adj_changes

def bisection(adj_changes, a, b, n_perturbations, epsilon):
    def func(x):
        return torch.clamp(adj_changes-x, 0, 1).sum() - n_perturbations

    miu = a
    while ((b-a) >= epsilon):
        miu = (a+b)/2
        # Check if middle point is root
        if (func(miu) == 0.0):
            break
        # Decide the side to repeat the steps
        if (func(miu)*func(a) < 0):
            b = miu
        else:
            a = miu
    # print("The value of root is : ","%.4f" % miu)
    return miu

