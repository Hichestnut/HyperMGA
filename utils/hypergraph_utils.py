# --------------------------------------------------------
# Utility functions for Hypergraph
#
# Author: Yifan Feng
# Date: November 2018
# --------------------------------------------------------
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split
import torch.sparse as ts
import torch.nn.functional as F
import warnings
import cvxpy as cp
from cvxpy.error import SolverError
from sklearn.cluster import KMeans
import scipy.sparse as sparse

def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list
    :param normal_col: normalize each column of the feature
    :return: Fused feature matrix
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    invDE = np.nan_to_num(invDE, nan=0, neginf=0)
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    DV2 = np.nan_to_num(DV2, nan=0, neginf=0)
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def normalize_H_tensor(H):
    """Normalize adjacency tensor matrix.
    """
    # device = adj.device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_edge = H.shape[1]
    G = torch.Tensor(H).to(device)

    # the weight of the hyperedge
    W = torch.ones(n_edge)
    # the degree of the node
    dv = G * W
    # print(dv)
    DV = dv.sum(1)
    # the degree of the hyperedge
    DE = G.sum(0)
    invDE = DE.pow(-1).flatten()
    invDE[torch.isinf(invDE)] = 0.
    mat_invDE = torch.diag(invDE)
    invDV = DV.pow(-1/2).flatten()
    invDV[torch.isinf(invDV)] = 0.
    mat_invDV = torch.diag(invDV)
    # DV2 = torch.tensor(torch.diag(torch.pow(torch.tensor([DV]), torch.tensor([-0.5]))), device='cpu')
    # DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    # H = np.mat(H)
    W = torch.diag(W)
    HT = G.t()
    # print(HT)
    # G = DV2 * H * W * invDE * HT * DV2
    MX = mat_invDV @ G @ W @ mat_invDE @ HT @ mat_invDV
    # MX = mat_invDV * G * W * mat_invDE * HT * mat_invDV
    return MX
    # mx = adj + torch.eye(adj.shape[0]).to(device)
    # rowsum = mx.sum(1)
    # r_inv = rowsum.pow(-1/2).flatten()
    # r_inv[torch.isinf(r_inv)] = 0.
    # r_mat_inv = torch.diag(r_inv)
    # mx = r_mat_inv @ mx
    # mx = mx @ r_mat_inv



def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=False, m_prob=1):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H

def construct_H_with_epsilonball(X, ratio):
    n_nodes = X.shape[0]
    n_edges = n_nodes
    H = np.zeros((n_nodes, n_edges))
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    m_dist = Eu_dis(X)
    # print(m_dist)
    avg_dist = np.mean(m_dist)
    threshold = ratio * avg_dist
    # print(avg_dist)
    # print(threshold)

    for i in range(n_nodes):
        for j in range(n_nodes):
            if m_dist[i,j] <= threshold:
                H[j][i] = 1
    # print(H)
    return H

def gen_l1_hg(X, gamma, n_neighbors, log=False, with_feature=False):
    """
    :param X: numpy array, shape = (n_samples, n_features)
    :param gamma: float, the tradeoff parameter of the l1 norm on representation coefficients
    :param n_neighbors: int,
    :param log: bool
    :param with_feature: bool, optional(default=False)
    :return: instance of HyperG
    """
    # assert n_neighbors >= 1.
    # assert isinstance(X, np.ndarray)
    # assert X.ndim == 2

    n_nodes = X.shape[0]
    n_edges = n_nodes
    H = np.zeros((n_nodes, n_edges))
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])

    m_dist = Eu_dis(X)
    m_neighbors = np.argsort(m_dist)[:, 0:n_neighbors+1]
    print(m_neighbors)
    edge_idx = np.tile(np.arange(n_edges).reshape(-1, 1), (1, n_neighbors+1)).reshape(-1)
    print(edge_idx)
    print(edge_idx.shape)
    node_idx = []
    values = []
    # X = np.array(X)
    print(X.shape)

    for i_edge in range(n_edges):

        neighbors = m_neighbors[i_edge].tolist()
        neighbors = neighbors[0]
        if i_edge in neighbors:
            neighbors.remove(i_edge)
        else:
            neighbors = neighbors[:-1]
        # print(neighbors)
        P = X[neighbors, :]
        v = X[i_edge, :]

        # cvxpy
        x = cp.Variable(P.shape[0], nonneg=True)
        objective = cp.Minimize(cp.norm((P.T@x).T-v, 2) + gamma * cp.norm(x, 1))
        # objective = cp.Minimize(cp.norm(x@P-v, 2) + gamma * cp.norm(x, 1))
        prob = cp.Problem(objective)
        try:
            prob.solve()
        except SolverError:
            prob.solve(solver='SCS', verbose=False)

        node_idx.extend([i_edge] + neighbors)
        values.extend([1.] + x.value.tolist())

    node_idx = np.array(node_idx)
    print(node_idx.shape)
    values = np.array(values)
    np.savetxt('values.txt', values)
    print(values.shape)
    print(node_idx.shape[0])
    print(values.shape[0])
    # values_ner = values - np.eye(values)
    # v_mean = np.mean(values)
    v_mean = (values.sum() - 2708) / (int(values.shape[0]) - 2708)
    v_mean = v_mean
    print(v_mean)
    for i in range(node_idx.shape[0]):
        node_i = node_idx[i]
        edge_i = edge_idx[i]
        if values[i] > v_mean:
            H[node_i][edge_i] = 1
        else:
            H[node_i][edge_i] = 0

    # H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))
    print(H)
    return H

def gen_clustering_hg(X, n_clusters, method="kmeans",random_state=None):

    if method == "kmeans":
        cluster = KMeans(n_clusters=n_clusters, random_state=random_state).fit(X).labels_
    else:
        raise ValueError("{} method is not supported".format(method))

    n_edges = n_clusters
    n_nodes = X.shape[0]
    H = np.zeros((n_nodes, n_edges))
    node_idx = np.arange(n_nodes)
    edge_idx = cluster

    values = np.ones(node_idx.shape[0])
    # H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))
    for i in range(node_idx.shape[0]):
        node_i = node_idx[i]
        edge_i = edge_idx[i]
        H[node_i][edge_i] = values[i]
    return H